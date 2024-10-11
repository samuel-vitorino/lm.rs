import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-id", type=str, required=True, help="model id for the corresponding tokenizer")
    parser.add_argument("-t", "--tokenizer-type", type=str, required=True, choices=['LLAMA', 'GEMMA', 'PHI'], help="type of tokenizer (GEMMA/LLAMA/PHI)")
    args = parser.parse_args()

    if args.tokenizer_type == "GEMMA":        
        from utils.tokenizers.gemma import Tokenizer
    elif args.tokenizer_type == "LLAMA":
        from utils.tokenizers.llama import Tokenizer
    elif args.tokenizer_type == "PHI":
        from utils.tokenizers.phi import Tokenizer
    
    t = Tokenizer(args.model_id)
    t.export()