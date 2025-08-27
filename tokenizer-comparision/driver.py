from tokenizers.bpe_tokenizer import BPETokenizer
import os
import click
from tokenizers.comparision import TokenizerComparision

@click.group()
def tokenizer():
    "Tokenizer CLI Tool"
    pass

@click.command()
@click.option('--vocab_size', default=100, help="Vocab size for the BPE model")
def train_bpe(vocab_size):
    if not os.path.exists(f'./models/bpe_sample_{vocab_size}.json'):
        print("Training sample model")
        sample = BPETokenizer(f'Sample {vocab_size} Vocab BPE Model', vocab_size=vocab_size)
        with open('./data/train.txt', 'r') as file:
            text = file.read()
        
        sample.train(text)
        sample.save_model_to_file(f'./models/bpe_sample_{vocab_size}.json')

    else:
        print("Loading sample model")
        sample = BPETokenizer('Sample {vocab_size} Vocab BPE Model', vocab_size=vocab_size)
        sample.load_model_from_file(os.path.join(os.path.dirname(__file__), 'models', f'bpe_sample_{vocab_size}.json'))

    test_str = ""
    print("="*20)
    print(f'Test String: {test_str}')
    print(f'Encoded String: {sample.encode(test_str)}')
    print(f'Decoded String: {sample.decode(sample.encode(test_str))}')


@click.command()
def compare_trained_bpe_models():
    # Load all the models present in ./models dir with prefix bpe
    tokenizers = []
    for file in os.listdir(os.path.join(os.path.dirname(__file__), 'models')):
        if file.startswith('bpe_sample_'):
            tokenizers.append(BPETokenizer())
            tokenizers[-1].load_model_from_file(os.path.join(os.path.dirname(__file__), 'models', file))

    comparision = TokenizerComparision(tokenizers)
    with open('./data/test.txt', 'r') as file:
        test_str = file.read()
    comparision.print_comparisions(comparision.compare_tokenization_results(test_str), "Tokenization Results")


tokenizer.add_command(train_bpe)
tokenizer.add_command(compare_trained_bpe_models)
    
if __name__ == '__main__':
    tokenizer()