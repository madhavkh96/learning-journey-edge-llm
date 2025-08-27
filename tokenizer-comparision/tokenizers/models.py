from abc import ABC, abstractmethod
import time
import json

class Tokenizer(ABC):
    def __init__(self, name: str | None = None):
        self.name = name
        self.vocabulary = []

    @abstractmethod
    def train(self, text: str):
        pass
    
    @abstractmethod
    def encode(self, text: str):
        pass
    
    @abstractmethod
    def decode(self, tokens: list[int]):
        pass
    
    @abstractmethod
    def save_model_to_file(self, file_path: str):
        pass
    
    @abstractmethod
    def load_model_from_file(self, file_path: str):
        pass
 
    def get_vocab_size(self) -> int:
        return len(self.vocabulary)

    def get_token_count(self, text: str) -> int:
        return len(self.encode(text))

    def get_compression_ratio(self, text: str) -> float:
        return 1 - (self.get_token_count(text) / len(text))

    def get_tokenization_time(self, text: str) -> float:
        start_time = time.time()
        self.encode(text)
        end_time = time.time()
        return end_time - start_time
    
 