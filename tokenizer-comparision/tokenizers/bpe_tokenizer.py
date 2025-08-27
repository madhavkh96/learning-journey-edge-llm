from .models import Tokenizer
from collections import Counter, defaultdict
import json

"""
ALGORITHM 
-   There is a pretokenization step that space tokenizes the training data into words.
-   BPE create a base vocabulary consisting of all symbols that occur in the set of unique words.
-   Now we merge two symbols based on which symbol pair occurs the most time together.
-   It does so until the desired vocabulary size has been attained.


TIME COMPLEXITY: 
-   O(n^2) -> Naive
-   TODO: O(nlogm) -> Optimized with Heap
"""


class BPETokenizer(Tokenizer):
    def __init__(self, name: str | None = None, vocab_size: int | None = None):
        super().__init__(name)
        self.vocab_size = vocab_size
        self.merge_rules = []
        self.vocab_list = []
        self.token_to_id = {}

    def train(self, text: str):

        if not self.vocab_size:
            raise ValueError("Vocab size can't be none.")
        
        # Pre tokenization step 
        pre_tokenized_words = defaultdict(int)

        for word in text.split():
            pre_tokenized_words[word] += 1            

        # Create a base vocabulary of single characters
        self.vocabulary = set("".join(pre_tokenized_words.keys()))

        # Represent each unique word as a list of symbols (characters)
        words_as_symbols = {word: list(word) for word in pre_tokenized_words.keys()}

        # Keep merging most frequent adjacent pairs until we hit desired vocab size
        while len(self.vocabulary) < self.vocab_size:
            pair_frequencies: Counter[tuple[str, str]] = Counter()

            # Count adjacent pair frequencies weighted by word occurrence
            for word, freq in pre_tokenized_words.items():
                symbols = words_as_symbols[word]
                if len(symbols) < 2:
                    continue
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    pair_frequencies[pair] += freq

            if not pair_frequencies:
                break

            # Select the most frequent pair to merge
            best_pair = max(pair_frequencies, key=pair_frequencies.get)
            merged_token = "".join(best_pair)

            # If merged token already in vocab and no further merges change anything, stop
            if merged_token in self.vocabulary and pair_frequencies[best_pair] == 0:
                break

            # Store the merge rule
            self.merge_rules.append(best_pair)

            # Apply the merge to all words' symbol lists
            for word in words_as_symbols.keys():
                symbols = words_as_symbols[word]
                if len(symbols) < 2:
                    continue
                merged_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == best_pair[0] and symbols[i + 1] == best_pair[1]:
                        merged_symbols.append(merged_token)
                        i += 2
                    else:
                        merged_symbols.append(symbols[i])
                        i += 1
                words_as_symbols[word] = merged_symbols

            # Add the new merged token to the vocabulary
            self.vocabulary.add(merged_token)

        # Build ordered vocabulary and token-to-id mapping
        self.vocab_list = sorted(list(self.vocabulary))
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab_list)}
        
        return pre_tokenized_words

    def encode(self, text: str):
        tokens = []
        text = text.replace(' ', '_') # Using _ as a space token similar to SentencePiece
        for word in text.split():
            # Start with individual characters
            word_tokens = list(word)
            
            # Apply all learned merge rules
            for pair in self.merge_rules:
                merged = "".join(pair)
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == pair[0] and word_tokens[i + 1] == pair[1]:
                        word_tokens[i:i+2] = [merged]
                    else:
                        i += 1
            
            # Convert tokens to ids
            for token in word_tokens:
                if token in self.token_to_id:
                    tokens.append(self.token_to_id[token])
                else:
                    # Handle unknown tokens (could use a special UNK token)
                    tokens.append(-1)  # or some other unknown token id
        
        return tokens

    def decode(self, tokens: list[int]):
        text = ""
        for token_id in tokens:
            if 0 <= token_id < len(self.vocab_list):
                text += self.vocab_list[token_id]
            else:
                # Handle unknown token ids
                text += "<UNK>"
        return text.replace('_', ' ')
    
    def save_model_to_file(self, file_path: str):
        with open(file_path, 'w') as file:
            json.dump({
                'name': self.name,
                'vocabulary': list(self.vocabulary),
                'merge_rules': self.merge_rules,
                'vocab_list': self.vocab_list,
                'token_to_id': self.token_to_id
            }, file)
    
    def load_model_from_file(self, file_path: str):
        with open(file_path, 'r') as file:
            data = json.load(file)
            self.name = data['name']
            self.vocabulary = set(data['vocabulary'])
            self.merge_rules = data['merge_rules']
            self.vocab_list = data['vocab_list']
            self.token_to_id = data['token_to_id']