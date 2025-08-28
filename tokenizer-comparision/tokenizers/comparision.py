from .models import Tokenizer
from typing import Dict
class TokenizerComparator:
    def __init__(self, tokenizers: list[Tokenizer]):
        self.tokenizers = tokenizers

    def compare_tokenization_results(self, text: str) -> Dict[str, str | float]:
        results = {
            "tokenizer": [],
            "vocab_size": [],
            "token_count": [],
            "raw_text_length": [],
            "compression_ratio": [],
        }
        for tokenizer in self.tokenizers:
            results["tokenizer"].append(tokenizer.name)
            results["vocab_size"].append(tokenizer.get_vocab_size())
            results["token_count"].append(tokenizer.get_token_count(text))
            results["raw_text_length"].append(len(text))
            results["compression_ratio"].append(tokenizer.get_compression_ratio(text))
        return results
    
    def compare_tokenization_time(self, path: str):
        results = {
            "tokenizer": [],
            "time": [],
        }
        for tokenizer in self.tokenizers:
            results["tokenizer"].append(tokenizer.name)
            results["time"].append(tokenizer.get_tokenization_time(path))
        return results
    
    def compare_training_time(self, path: str):
        results = {
            "tokenizer": [],
            "time": [],
        }

        for tokenizer in self.tokenizers:
            results["tokenizer"].append(tokenizer.name)
            results["time"].append(tokenizer.get_training_time(path))
        return results
    
    def print_comparisions(self, results: dict[str, list], title: str, print_header: bool = True):
        if print_header:
            print(title)

        divider = "|" + "="*35 + "="*22*(len(results.keys()) - 1)

        print(divider)

        for key in results.keys():
            if key == "tokenizer":
                print(f"|{key:<35}|", end="")
            else:
                print(f"{key:<20}|", end="")
        print()
        print(divider)
        for i in range(len(results["tokenizer"])):
            for key in results.keys():
                if key == "compression_ratio":
                    print(f"{results[key][i]:<20.2f}|", end="")
                elif key == "time":
                    print(f"{results[key][i]:<20.4f}|", end="")
                elif key == "tokenizer":
                    print(f"|{results[key][i]:<35}|", end="")
                else:
                    print(f"{results[key][i]:<20}|", end="")
            print()
        print(divider)

    
