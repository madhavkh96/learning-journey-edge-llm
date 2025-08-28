from abc import ABC, abstractmethod
import time

class Tokenizer(ABC):
    def __init__(self, name: str | None = None, vocab_size: int = 100):
        self.name = name
        self.vocab_size = vocab_size
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

    def _load_file_from_path(self, path: str) -> str:
        with open(path, 'r') as file:
            text = file.read()
        return text
 
    def get_vocab_size(self) -> int:
        return self.vocab_size

    def get_token_count(self, text: str) -> int:
        return len(self.encode(text))

    def get_compression_ratio(self, text: str) -> float:
        return 1 - (self.get_token_count(text) / len(text))

    def get_tokenization_time(self, path: str) -> float:
        text = self._load_file_from_path(path)
        start_time = time.time()
        self.encode(text)
        end_time = time.time()
        return end_time - start_time
    
    def get_training_time(self, path: str) -> float:
        start_time = time.time()
        self.train(path)
        end_time = time.time()
        return end_time - start_time