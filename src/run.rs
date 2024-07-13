mod functional;

fn main() {
    let mut a: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 6.0];

    functional::softmax(&mut a);

    for i in a.iter() {
        println!("{}", i)
    }
}