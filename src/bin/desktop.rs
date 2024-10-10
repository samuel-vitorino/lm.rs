use dioxus::prelude::*;

use dioxus_desktop::Config;
use lmrs::sampler::Sampler;
use lmrs::tokenizer::Tokenizer;
use lmrs::transformer::ModelType;
use lmrs::transformer::Transformer;

use chrono::Local;
use clap::Parser;
use memmap2::Mmap;
use std::fs;
use std::fs::File;
use std::sync::mpsc::channel;
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: String,
    #[arg(long, default_value_t = String::from("tokenizer.bin"))]
    tokenizer: String,
    #[arg(long, default_value_t = 1.0f32)]
    temperature: f32,
    #[arg(long, default_value_t = 0.9f32)]
    top_p: f32,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, default_value_t = false)]
    show_metrics: bool,
}

enum ModelResponse {
    Piece(String),
    Termination,
}

struct Context<'a> {
    user_idx: usize,
    pos: u32,
    token: u32,
    next: u32,
    num_prompt_tokens: usize,
    total_tokens: f32,
    total_duration: f32,
    model: Transformer<'a>,
    prompt_tokens: Vec<u32>,
    tokenizer: Tokenizer,
    sampler: Sampler,
}

impl<'a> Context<'a> {
    fn handle_user_prompt(&mut self, user_prompt: &str) {
        // System prompt
        if self.model.args.model_type == ModelType::LLAMA && self.pos == 0 {
            // First part of chat template with initial tags and cut off date
            self.prompt_tokens.extend([
                128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366,
                18, 198, 15724, 2696, 25, 220,
            ]);

            let today = Local::now().date_naive();
            let formatted_date = today.format("%d %b %Y").to_string();
            self.prompt_tokens.extend(self.tokenizer.encode(
                &formatted_date,
                false,
                false,
                false,
                self.model.args.model_type,
            ));

            self.prompt_tokens.extend([271, 128009])
        }

        self.prompt_tokens.extend(self.tokenizer.encode(
            user_prompt.trim(),
            false,
            false,
            true,
            self.model.args.model_type,
        ));
        self.num_prompt_tokens = self.prompt_tokens.len();

        self.user_idx = 0;
    }

    fn compute_model_response(&mut self, show_metrics: bool) -> ModelResponse {
        loop {
            if self.user_idx < self.num_prompt_tokens {
                self.token = self.prompt_tokens[self.user_idx];
                self.user_idx += 1;
            } else {
                self.token = self.next;
            }

            if self.token == self.tokenizer.eos && self.user_idx >= self.num_prompt_tokens {
                self.prompt_tokens.clear();

                if show_metrics {
                    let toks = self.total_tokens / (self.total_duration / 1000.0);

                    println!("Speed: {:.2} tok/s", toks);

                    self.total_duration = 0.0;
                    self.total_tokens = 0.0;
                }
                return ModelResponse::Termination;
            }

            let processing_start = Instant::now();
            let logits: &mut [f32] = self.model.forward(self.token, self.pos);
            self.next = self.sampler.sample(logits);
            self.pos += 1;
            if self.user_idx >= self.num_prompt_tokens
                && self.next != self.tokenizer.eos
                && !(self.model.args.model_type == ModelType::GEMMA && self.next == 107)
            {
                let piece = self.tokenizer.decode(self.next);
                return ModelResponse::Piece(piece);
            }
            let duration = processing_start.elapsed();
            self.total_duration += duration.as_millis() as f32;
            self.total_tokens += 1.0;
        }
    }
}

fn main() {
    dioxus_desktop::launch::launch(app, Vec::new(), Config::default());
}

enum Message {
    Bot(String),
    User(String),
}

fn app() -> Element {
    let (to_model_sender, to_model_receiver) = channel::<String>();
    let to_model_channel = use_signal(move || to_model_sender);

    let mut user_input = use_signal(|| String::new());
    let mut is_ctrl_pressed = use_signal(|| false);
    let mut is_waiting_for_response = use_signal_sync(|| false);

    let mut conversation = use_signal_sync(|| Vec::<Message>::new());

    use_hook(|| {
        thread::spawn(move || loop {
            let args = Args::parse();
            let model_path: &str = args.model.as_str();
            let tokenizer_path: &str = args.tokenizer.as_str();

            assert!(
                fs::metadata(tokenizer_path).is_ok(),
                "Tokenizer file not found: {}",
                tokenizer_path
            );
            assert!(
                fs::metadata(model_path).is_ok(),
                "Model file not found: {}",
                model_path
            );

            let tokenizer = Tokenizer::new(args.tokenizer.as_str());

            let file = File::open(model_path).expect("Error opening model file");
            let data = unsafe { Mmap::map(&file).expect("MMap failed") };

            let model = Transformer::new(&data);

            let seed: u64 = match args.seed {
                Some(seed_value) => seed_value,
                None => {
                    let start = SystemTime::now();
                    let since_epoch = start
                        .duration_since(UNIX_EPOCH)
                        .expect("Error getting time since epoch");

                    since_epoch.as_millis() as u64
                }
            };

            let sampler = Sampler::new(model.args.vocab_size, args.temperature, args.top_p, seed);

            let mut context = Context {
                user_idx: 0,
                pos: 0,
                token: 0,
                next: 0,
                num_prompt_tokens: 0,
                total_tokens: 0.0,
                total_duration: 0.0,
                model,
                prompt_tokens: Vec::new(),
                tokenizer,
                sampler,
            };

            loop {
                let Ok(user_prompt) = to_model_receiver.recv() else {
                    // The sender is droped when the app is closed
                    break;
                };

                context.handle_user_prompt(&user_prompt);

                conversation.write().push(Message::Bot(String::new()));

                loop {
                    let ModelResponse::Piece(resp_piece) = context.compute_model_response(true)
                    else {
                        is_waiting_for_response.set(false);
                        break;
                    };
                    let mut conv = conversation.write();
                    let Message::Bot(response) = conv.last_mut().expect("The bot response exists")
                    else {
                        panic!("The user cannot add messages to the conversation while the model is processing its response");
                    };
                    response.push_str(&resp_piece);
                }
            }
        });
    });

    let mut handle_submit = {
        let mut user_input = user_input.clone();

        move || {
            if is_waiting_for_response() {
                // do nothing when the model is already processing the response
                return;
            }

            let input_string = user_input();
            conversation
                .write()
                .push(Message::User(input_string.clone()));
            let input_trimmed = input_string.trim().to_owned();
            if input_trimmed.is_empty() {
                return;
            }
            is_waiting_for_response.set(true);
            user_input.write().clear();
            to_model_channel()
                .send(input_trimmed)
                .expect("Failed to send the input to the model thread");
        }
    };

    rsx! {
        head {
            link {
                rel: "stylesheet",
                href: "https://samuel-vitorino.github.io/lm.rs-webui/assets/index-DTGPVQCS.css",
            }
        }
        body {
            class: "dark:bg-[#212121]",
            div {
                id: "root",
                class: "flex flex-col",
                nav {
                    class: "bg-white px-2 py-2.5 dark:border-gray-700 sm:px-4 dark:bg-[#212121]",
                    div {
                        class: "mx-auto flex flex-wrap items-center justify-between",
                        span {
                            class: "flex items-center",
                            img {
                                src: "https://samuel-vitorino.github.io/lm.rs-webui/rust.svg",
                                class: "mr-3 h-6 sm:h-9",
                                alt: "lm.rs logo",
                            }
                            span {
                                class: "self-center whitespace-nowrap text-xl font-semibold dark:text-white",
                                "lm.rs",
                            }
                        }
                    }
                }
                div {
                    class: "flex flex-col justify-between mx-3 sm:mx-20 md:mx-44 xl:mx-96",
                    style: "max-height: 78%; overflow-y: auto;",
                    id: "chat-container",
                    for message in conversation.read().iter(){
                        {
                            match message {
                                Message::Bot(msg) => {
                                    rsx!{ MessageComponent{
                                        message: msg,
                                        is_user: false,
                                    }}
                                },
                                Message::User(msg) => {
                                    rsx!{ MessageComponent{
                                        message: msg,
                                        is_user: true,
                                    }}
                                },
                            }
                        }
                    }
                }
                footer {
                    class: "flex items-center justify-around dark:bg-[#303030] dark:text-white bg-[#f5f5f5] mx-3 md:m-0 md:w-6/12 md:self-center",
                    id: "chat-input",
                    textarea {
                        rows: "1",
                        placeholder: "Message lm.rs",
                        class: "bg-transparent",
                        id: "chat-text-area",
                        disabled: is_waiting_for_response(),
                        value: "{user_input}",
                        oninput: move |e| user_input.set(e.value().clone()),
                        onkeydown: move|event|{
                            if event.key() == Key::Control{
                                is_ctrl_pressed.set(true);
                            }
                        },
                        onkeyup: move|event|{
                            if event.key() == Key::Control{
                                is_ctrl_pressed.set(false);
                            }
                        },
                        onkeypress: move |event| {
                            if event.key() == Key::Enter && is_ctrl_pressed(){
                                handle_submit()
                            }
                        },
                    }
                    button {
                        id: "chat-send-button",
                        class: "flex",
                        disabled: is_waiting_for_response(),
                        onclick: move |_| handle_submit(),
                        svg {
                            stroke: "currentColor",
                            fill: "currentColor",
                            "stroke-width": "0",
                            view_box: "0 0 448 512",
                            class: "self-center",
                            height: "20px",
                            width: "20px",
                            xmlns: "http://www.w3.org/2000/svg",
                            path {
                                d: "M34.9 289.5l-22.2-22.2c-9.4-9.4-9.4-24.6 0-33.9L207 39c9.4-9.4 24.6-9.4 33.9 0l194.3 194.3c9.4 9.4 9.4 24.6 0 33.9L413 289.4c-9.5 9.5-25 9.3-34.3-.4L264 168.6V456c0 13.3-10.7 24-24 24h-32c-13.3 0-24-10.7-24-24V168.6L69.2 289.1c-9.3 9.8-24.8 10-34.3.4z"
                            }
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn MessageComponent(message: String, is_user: bool) -> Element {
    rsx! {
        div {
            class: if is_user {"flex items-center message_l1_user" } else { "flex items-center message_l1_bot"},
            style: if is_user {r#"
                align-self: flex-end;
                margin-top: 0px;
                margin-bottom: 0px;
                width: 100%;
            "# }else{ r#"
                align-self: flex-start;
                margin-top: 30px;
                margin-bottom: 30px;
                width: 100%;
            "#},
            div {
                class: "flex flex-col",
                    style: if is_user {r#"
                        margin-left: 0px;
                        text-wrap: wrap;
                        overflow-wrap: break-word;
                        width: 100%;
                    "# }else{ r#"
                        margin-left: 20px;
                        text-wrap: wrap;
                        overflow-wrap: break-word;
                        width: 100%;
                    "#},
                Markdown {
                    class: if is_user { "bg-[#f5f5f5] dark:bg-[#323232] dark:text-white" } else {"dark:text-white"},
                    style: if is_user { r#"
                        border-radius: 20px;
                        padding: 10px 20px 10px 20px;
                        max-width: 40%;
                        margin-left: auto;
                        width: fit-content;
                    "#}else{r#"
                        margin-left: 0px;
                        text-wrap: wrap;
                        overflow-wrap: break-word;
                        width: 100%;
                    "#},
                    input: "{message}"
                }
            }
        }
    }
}

#[component]
fn Markdown(input: String, class: String, style: String) -> Element {
    let parser = pulldown_cmark::Parser::new(&input);
    let mut html_output: String = String::with_capacity(input.len() * 3 / 2);
    pulldown_cmark::html::push_html(&mut html_output, parser);

    rsx! { div {
    class: "{class}",
    style: "{style}",
    dangerous_inner_html: "{html_output}" } }
}
