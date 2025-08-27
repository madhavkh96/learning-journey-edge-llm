from tokenizers.bpe_tokenizer import BPETokenizer
import os
import click
from tokenizers.comparision import TokenizerComparator

@click.group()
def tokenizer():
    "Tokenizer CLI Tool"
    pass

@click.command()
@click.option('--vocab_size', default=100, help="Vocab size for the BPE model")
def train_bpe(vocab_size):
    if not os.path.exists(f'./models/bpe_{vocab_size}.json'):
        print("Training BPE model")
        model = BPETokenizer(f'{vocab_size} Vocab Size BPE Model', vocab_size=vocab_size)
        with open('./data/train.txt', 'r') as file:
            text = file.read()
        
        model.train(text)
        model.save_model_to_file(f'./models/bpe_{vocab_size}.json')

    else:
        print("Loading BPE model")
        model = BPETokenizer('{vocab_size} Vocab Size BPE Model', vocab_size=vocab_size)
        model.load_model_from_file(os.path.join(os.path.dirname(__file__), 'models', f'bpe_{vocab_size}.json'))

    test_str = "Hello, world! I am going to the store and my name is Grady!"
    print("="*20)
    print(f'Test String: {test_str}')
    print(f'Encoded String: {model.encode(test_str)}')
    print(f'Decoded String: {model.decode(model.encode(test_str))}')


@click.command()
def compare_trained_bpe_models():
    # Load all the models present in ./models dir with prefix bpe
    tokenizers = []
    for file in os.listdir(os.path.join(os.path.dirname(__file__), 'models')):
        if file.startswith('bpe_'):
            tokenizers.append(BPETokenizer())
            tokenizers[-1].load_model_from_file(os.path.join(os.path.dirname(__file__), 'models', file))

    comparision = TokenizerComparator(tokenizers)
    with open('./data/test.txt', 'r') as file:
        test_str = file.read()
    comparision.print_comparisions(comparision.compare_tokenization_results(test_str), "Tokenization Results")


tokenizer.add_command(train_bpe)
tokenizer.add_command(compare_trained_bpe_models)
    
if __name__ == '__main__':
    tokenizer()