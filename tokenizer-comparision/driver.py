from tokenizers.bpe_tokenizer import BPETokenizer
from tokenizers.sentencepiece_wrapper import SentencePieceWrapper
import os
import click
from typing import List
from tokenizers.models import Tokenizer
from tokenizers.comparision import TokenizerComparator

@click.group()
def tokenizer():
    "Tokenizer CLI Tool"
    pass

@click.command()
@click.option('--vocab-size', default=100, help="Vocab size for the BPE model")
def train_bpe(vocab_size):
    if not os.path.exists(f'./models/bpe_{vocab_size}.json'):
        print("Training BPE model")
        model = BPETokenizer(f'{vocab_size} Vocab Size BPE Model', vocab_size=vocab_size)
        model.train('./data/train.txt')
        model.save_model_to_file(f'./models/bpe_{vocab_size}.json')

    else:
        print("Loading BPE model")
        model = BPETokenizer('{vocab_size} Vocab Size BPE Model', vocab_size=vocab_size)
        model.load_model_from_file(os.path.join(os.path.dirname(__file__), 'models', f'bpe_{vocab_size}.json'))

    test_str = "Hello, world! I am going to the store and my name is Grady!"
    print("="*40)
    print(f'Test String: {test_str}')
    print(f'Encoded String: {model.encode(test_str)}')
    print(f'Decoded String: {model.decode(model.encode(test_str))}')

@click.command()
@click.option('--vocab-size', default=100, help="Vocab size for the SentencePiece model")
@click.option('--model-type', default='bpe', help="Model type that the SentencePiece model should use")
@click.option('--model-prefix', default='spm_bpe', help="Model file name prefix for saving")
def train_sentencepiece(vocab_size, model_type, model_prefix):
    model_prefix = f'{model_prefix}_{vocab_size}'
    if not os.path.exists(f'./models/{model_prefix}.model'):
        model = SentencePieceWrapper(model_prefix, vocab_size)
        model.train('./data/train.txt')
        
        test_str = "Hello, world! I am going to the store and my name is Grady!"
        print("="*40)
        print(f'Test String: {test_str}')
        print(f'Encoded String: {model.encode(test_str)}')
        print(f'Decoded String: {model.decode(model.encode(test_str))}')


def __load_tokenizers() -> List[Tokenizer]:
    tokenizers: List[Tokenizer] = []
    for file in os.listdir(os.path.join(os.path.dirname(__file__), 'models')):
        if file.endswith('.vocab'):
            continue
        if file.startswith('custom'):
            model_name = file.split('.')[0]
            vocab_size = model_name.split('_')[2]
            tokenizers.append(BPETokenizer(model_name, int(vocab_size)))
            tokenizers[-1].load_model_from_file(os.path.join(os.path.dirname(__file__), 'models', file))
        elif file.startswith('spm'):
            model_name = file.split('.')[0]
            vocab_size = model_name.split('_')[2]
            tokenizers.append(SentencePieceWrapper(model_name, int(vocab_size)))
            tokenizers[-1].load_model_from_file(f'./models/{file}')
    return tokenizers

@click.command()
def compare_tokenization_bpe_models():
    # Load all the models present in ./models dir with prefix bpe
    tokenizers: List[Tokenizer] = __load_tokenizers()
    comparators = TokenizerComparator(tokenizers)
    with open('./data/test.txt', 'r') as file:
        test_str = file.read()
    results = comparators.compare_tokenization_results(test_str)
    comparators.print_comparisions(results, "Tokenization Results")

@click.command()
@click.option('--vocab_size', default='100', help='Vocab size for each model')
def compare_training_time_models(vocab_size):
    tokenizers = []
    for name in [f'custom_bpe_{vocab_size}', f'spm_bpe_{vocab_size}']:
        if 'custom_bpe' in name:
            tokenizers.append(BPETokenizer(name, vocab_size))
        elif 'spm_bpe' in name:
            tokenizers.append(SentencePieceWrapper(name, vocab_size))
        else:
            NotImplementedError(f"Currently Tokenizer {name} not supported")
    comparator = TokenizerComparator(tokenizers)
    results = comparator.compare_tokenization_time('./data/train.txt')
    comparator.print_comparisions(results, "Training Time Results")

tokenizer.add_command(train_bpe)
tokenizer.add_command(compare_tokenization_bpe_models)
tokenizer.add_command(train_sentencepiece)
tokenizer.add_command(compare_training_time_models)
    
if __name__ == '__main__':
    tokenizer()