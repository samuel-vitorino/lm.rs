import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--tokenizer-model", type=str, required=True, help="optional path to custom tokenizer ")
    parser.add_argument("-t", "--tokenizer-type", type=str, required=True, choices=['LLAMA', 'GEMMA'], help="optional path to custom tokenizer ")
    args = parser.parse_args()

    if args.tokenizer_type == "GEMMA":        
        from common.tokenizers.gemma import Tokenizer
    elif args.tokenizer_type == "LLAMA":
        from common.tokenizers.llama import Tokenizer
    
    t = Tokenizer(args.tokenizer_model)
    t.export()