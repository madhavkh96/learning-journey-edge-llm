import sentencepiece as spm
from .models import Tokenizer

class SentencePieceWrapper(Tokenizer):
    def __init__(self, name: str | None = None, vocab_size: int | None = None, model_type: str = 'bpe', persist: bool = True):
        super().__init__(name, vocab_size)
        self.model_type = model_type
        self.vocabulary = []
        self.sp = spm.SentencePieceProcessor()
        self.persist = persist
        if not self.name:
            raise ValueError("Model Name not provided")
        self.model_file_path = f'./models/{self.name}.model'
        
    def train(self, path: str):
        if self.persist:
            spm.SentencePieceTrainer.Train(f'--input={path} --model_prefix=./models/{self.name} --vocab_size={self.vocab_size} --model_type={self.model_type}')
        else:
            spm.SentencePieceTrainer.Train(f'--input={path} --vocab_size={self.vocab_size} --model_type={self.model_type}')
            
    def encode(self, text: str):
        self.load_model_from_file(self.model_file_path)
        return self.sp.EncodeAsIds(text)
    
    def decode(self, tokens: list[int]):
        return self.sp.DecodeIds(tokens)
    
    def save_model_to_file(self, file_path: str):
        pass
    
    def load_model_from_file(self, file_path: str):
        return self.sp.Load(file_path)