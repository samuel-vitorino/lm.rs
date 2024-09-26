import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--tokenizer-model", type=str, required=True, help="path to custom tokenizer model file ")
    parser.add_argument("-t", "--tokenizer-type", type=str, required=True, choices=['LLAMA', 'GEMMA'], help="type of tokenizer (GEMMA/LLAMA)")
    args = parser.parse_args()

    if args.tokenizer_type == "GEMMA":        
        from common.tokenizers.gemma import Tokenizer
    elif args.tokenizer_type == "LLAMA":
        from common.tokenizers.llama import Tokenizer
    
    t = Tokenizer(args.tokenizer_model)
    t.export()