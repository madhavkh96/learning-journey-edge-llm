# Tokenizer Comparison Project

A comprehensive toolkit for training, comparing, and analyzing different tokenization algorithms, with a focus on Byte-Pair Encoding (BPE) implementation.

## 📁 Directory Structure

```
tokenizer-comparision/
├── README.md                 # This file
├── driver.py                 # CLI tool for training and comparing tokenizers
├── requirements.txt          # Python dependencies
├── Notes.md                  # Detailed notes on tokenization algorithms
├── data/                     # Training and test data
│   ├── train.txt             # Large training dataset (587KB)
│   └── test.txt              # Test dataset (31KB)
├── models/                   # Trained tokenizer models
│   ├── bpe_100.json          # BPE model with 100 vocab size
│   ├── bpe_500.json          # BPE model with 500 vocab size
│   ├── bpe_1000.json         # BPE model with 1000 vocab size
│   ├── bpe_5000.json         # BPE model with 5000 vocab size
│   └── bpe_10000.json        # BPE model with 10000 vocab size
├── tokenizers/               # Tokenizer implementations
│   ├── __init__.py
│   ├── models.py            # Base Tokenizer abstract class
│   ├── bpe_tokenizer.py     # BPE implementation
│   └── comparision.py       # Comparison utilities
├── notebooks/               # Jupyter notebooks for experiments
│   └── experiments.ipynb    # Interactive experiments
└── notes-assets/            # Assets for documentation
```

## 🚀 Quick Start

### Basic Usage

The project provides a CLI tool (`driver.py`) for training and comparing tokenizers:

```bash
# Train a BPE tokenizer with 1000 vocabulary size
python driver.py train-bpe --vocab_size 1000

# Compare all trained BPE models
python driver.py compare-trained-bpe-models
```

## 📈 Driver.py CLI Tool

The `driver.py` file provides a command-line interface for managing tokenizer training and comparison.

### Available Commands

#### 1. `train-bpe`

Trains a new BPE tokenizer with specified vocabulary size.

**Usage:**

```bash
python driver.py train-bpe --vocab_size 1000
```

**Options:**

- `--vocab_size`: Vocabulary size for the BPE model (default: 100)

**What it does:**

- Checks if a model with the specified vocab size already exists
- If not, trains a new BPE tokenizer on `./data/train.txt`
- Saves the trained model to `./models/bpe_{vocab_size}.json`
- Tests the model with a sample string and shows encoding/decoding results

**Example output:**

```
Training sample model
====================
Test String: Hello, world! I am going to the store and my name is Grady!
Encoded String: [23, 45, 12, 67, 89, ...]
Decoded String: Hello, world! I am going to the store and my name is Grady!
```

#### 2. `compare-trained-bpe-models`

Compares all trained BPE models in the `./models/` directory.

**Usage:**

```bash
python driver.py compare-trained-bpe-models
```

**What it does:**

- Loads all BPE models from `./models/` directory
- Compares their performance on a test string
- Displays comparison metrics including:
  - Vocabulary size
  - Token count
  - Compression ratio
  - Tokenization time

**Example output:**

```
Tokenization Results
|========================================================================================================================
|tokenizer                          |vocab_size          |token_count         |raw_text_length     |compression_ratio   |
|========================================================================================================================
|100 Vocab Size BPE Model           |100                 |28480               |31387               |0.09                |
|1000 Vocab Size BPE Model          |1000                |15640               |31387               |0.50                |
|500 Vocab Size BPE Model           |500                 |17119               |31387               |0.45                |
|10000 Vocab Size BPE Model         |10000               |12373               |31387               |0.61                |
|5000 Vocab Size BPE Model          |5000                |13168               |31387               |0.58                |
|------------------------------------------------------------------------------------------------------------------------
```

### CLI Structure

The tool uses Click library for command-line interface:

```python
@click.group()
def tokenizer():
    "Tokenizer CLI Tool"
    pass

@click.command()
@click.option('--vocab_size', default=100, help="Vocab size for the BPE model")
def train_bpe(vocab_size):
    # Training logic

@click.command()
def compare_trained_bpe_models():
    # Comparison logic

# Register commands
tokenizer.add_command(train_bpe)
tokenizer.add_command(compare_trained_bpe_models)
```

## 📊 Tokenizer Implementations

### BPE Tokenizer (`tokenizers/bpe_tokenizer.py`)

A custom implementation of Byte-Pair Encoding with the following features:

- **Training**: Learns merge rules from training data
- **Encoding**: Applies learned merges to tokenize new text
- **Decoding**: Reconstructs original text from tokens
- **Vocabulary Management**: Maintains ordered vocabulary with token-to-id mapping

**Key Methods:**

- `train(text)`: Trains the tokenizer on input text
- `encode(text)`: Converts text to token IDs
- `decode(tokens)`: Converts token IDs back to text

### Comparison Utilities (`tokenizers/comparision.py`)

Provides utilities for comparing different tokenizers:

- **Performance Metrics**: Vocabulary size, token count, compression ratio
- **Timing Analysis**: Tokenization speed comparison
- **Results Formatting**: Tabular output for easy analysis

## 📈 Experiments and Analysis

### Jupyter Notebooks (`notebooks/`)

The `experiments.ipynb` notebook provides an interactive environment for:

- Testing different tokenizer configurations
- Analyzing tokenization patterns
- Visualizing compression ratios
- Experimenting with different training data

### Pre-trained Models

The `models/` directory contains pre-trained BPE models with varying vocabulary sizes:

- **100 tokens**: Basic vocabulary for simple text
- **500 tokens**: Medium vocabulary for general use
- **1000 tokens**: Larger vocabulary for better compression
- **5000 tokens**: Large vocabulary for complex text
- **10000 tokens**: Very large vocabulary for maximum compression

## 📝 Notes and Documentation

- **`Notes.md`**: Comprehensive documentation on tokenization algorithms
- **`notes-assets/`**: Supporting materials and diagrams

## 🛠️ Development

### Adding New Tokenizers

To add a new tokenizer implementation:

1. Create a new file in `tokenizers/` (e.g., `wordpiece_tokenizer.py`)
2. Inherit from the base `Tokenizer` class in `models.py`
3. Implement the required methods: `train()`, `encode()`, `decode()`
4. Add the new tokenizer to the comparison utilities

### Extending the CLI

To add new commands to `driver.py`:

1. Define a new function with `@click.command()` decorator
2. Add the command to the group using `tokenizer.add_command()`
3. Update this README with usage instructions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

## 📄 License

[Add your license information here]

---

**Note**: This project is part of a learning journey focused on understanding and implementing edge LLM tokenization techniques. The implementations are educational and may not be optimized for production use.
