
# Table of contents

- [Tokenizers](#tokenizers)
- [Types of tokenizers](#types-of-tokenizers)
  - [Rule based tokenizers](#rule-based-tokenizers)
  - [Character based](#character-based)
  - [Subword Tokenization](#subword-tokenization)
- [Byte-Pair Encoding (BPE)](#byte-pair-encoding-bpe)
  - [Example](#example)
  - [Byte-level BPE](#byte-level-bpe)
- [WordPiece](#wordpiece)
- [Unigram](#unigram)
- [SentencePiece](#sentencepiece)
  - [About](#about)
  - [Paper Summary](#paper-summary)
  - [Characteristics of the Library](#characteristics-of-the-library)


# Tokenizers
Tokenizing a text is splitting it into words or subwords, which then are converted to ids through a look-up table. We will be focusing on three tokenizers:
- Byte-Pair Encoding (BPE)
- WordPiece
- SentencePiece

## Types of tokenizers

- Rule Based
- Character Based
- Sub-word tokenization

### Rule based tokenizers

The rule based tokenizers basically help us define specific rules on how to treat different characters/words while tokenizing a text for example.

for the sentence:
`"Don't you love Transformers? We sure do."`

different tokenizing strategies might be:

e.g.

```python
["Don't", "you", "love", "Transformers?", "We", "sure", "do."]
["Don", "'", "t", "you", "love", "Transformers", "?", "We", "sure", "do", "."]
["Do", "n't", "you", "love", "🤗", "Transformers", "?", "We", "sure", "do", "."]
```

The last one being the most similar to what a rule based tokenizer like `spaCy` or `Moses` would output

Pros:

- Intuitive.
- Can form strong and meaningful associations.

Cons:

- It generates a very big vocabulary. Thus forces model to have an enormous embedding matrix
- Increases space and time complexity.

### Character based

As the name suggests this is a character based tokenizers and limits tokens based on character limit.

e.g. `Tod` instead of `Today`.

Pros:

- Fast

Cons:

- Hard to create associations based on random words.
- Loss of performance

### Subword Tokenization

Subword tokenization rely on the principle that frequently used words should not be split into smaller subwords, but rare words should be decomposed into meaningful subwords.

e.g. `annoyingly` -> `annoying` + `ly`

`BertTorkenizer` tokenizes `"I have a new GPU!"` as follows

```python
>>> from transformers import BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> tokenizer.tokenize("I have a new GPU!")
["i", "have", "a", "new", "gp", "##u", "!"]
```

Because we are considering the uncased model, the sentence was lowercased first. We can see that the words ["i", "have", "a", "new"] are present in the tokenizer’s vocabulary, but the word "gpu" is not. Consequently, the tokenizer splits "gpu" into known subwords: ["gp" and "##u"]. "##" means that the rest of the token should be attached to the previous one, without space (for decoding or reversal of the tokenization).

Another example

```python
>>> from transformers import XLNetTokenizer

>>> tokenizer = XLNetTokenizer.from_pretrained("xlnet/xlnet-base-cased")
>>> tokenizer.tokenize("Don't you love 🤗 Transformers? We sure do.")
["▁Don", "'", "t", "▁you", "▁love", "▁", "🤗", "▁", "Transform", "ers", "?", "▁We", "▁sure", "▁do", "."]
```

## Byte-Pair Encoding (BPE)

BPE relies on a pre-tokenizer that splits the training data into words. This can be as simple as space tokenization. After pre-tokenization, a set of unique words has been created and the frequency with which each word occured in the training data has been determined. Next, BPE create a base vocabulary consisting of all symbols that occur in the set of unique words and learns merge rules to form a new symbol from teo symbols of the base vocabulary. It does so until the vocab has attained the desired vocab size. *NOTE: The desired vocab size is a hyperparameter to define before training the tokenizer*

#### Example

As an example, let’s assume that after pre-tokenization, the following set of words including their frequency has been determined:

```
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

Consqeuntly our base vocabulary is
`["b", "g", "h", "n", "p", "s", "u"]`

Splitting all words into symbols of the base vocabulary, we obtain:

```
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```

BPE then counts the frequency of each possible symbol pair and picks the symbol pair that occurs most frequently. In the example above "h" followed by "u" is present 10 + 5 = 15 times (10 times in the 10 occurrences of "hug", 5 times in the 5 occurrences of "hugs"). However, the most frequent symbol pair is "u" followed by "g", occurring 10 + 5 + 5 = 20 times in total. Thus, the first merge rule the tokenizer learns is to group all "u" symbols followed by a "g" symbol together. Next, "ug" is added to the vocabulary. The set of words then becomes

```
("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
```

BPE then identifies the next most common symbol pair. It’s "u" followed by "n", which occurs 16 times. "u", "n" is merged to "un" and added to the vocabulary. The next most frequent symbol pair is "h" followed by "ug", occurring 15 times. Again the pair is merged and "hug" can be added to the vocabulary.

At this stage, the vocabulary is `["b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug"]` and our set of unique words is represented as

```
("hug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("hug" "s", 5)
```

Assuming, that the Byte-Pair Encoding training would stop at this point, the learned merge rules would then be applied to new words (as long as those new words do not include symbols that were not in the base vocabulary). For instance, the word "bug" would be tokenized to ["b", "ug"] but "mug" would be tokenized as ["<unk>", "ug"] since the symbol "m" is not in the base vocabulary. In general, single letters such as "m" are not replaced by the "<unk>" symbol because the training data usually includes at least one occurrence of each letter, but it is likely to happen for very special characters like emojis.

### Byte-level BPE

A base vocabulary that includes all possible base characters can be quite large if e.g. all unicode characters are considered as base characters. To have a better base vocabulary, GPT-2 uses bytes as the base vocabulary, which is a clever trick to force the base vocabulary to be of size 256 while ensuring that every base character is included in the vocabulary. With some additional rules to deal with punctuation, the GPT2’s tokenizer can tokenize every text without the need for the <unk> symbol. GPT-2 has a vocabulary size of 50,257, which corresponds to the 256 bytes base tokens, a special end-of-text token and the symbols learned with 50,000 merges.

## WordPiece

WordPiece is the subword tokenization algorithm used for BERT, DistilBERT, and Electra. The algorithm was outlined in Japanese and Korean Voice Search (Schuster et al., 2012) and is very similar to BPE. WordPiece first initializes the vocabulary to include every character present in the training data and progressively learns a given number of merge rules. In contrast to BPE, WordPiece does not choose the most frequent symbol pair, but the one that maximizes the likelihood of the training data once added to the vocabulary.

So what does this mean exactly? Referring to the previous example, maximizing the likelihood of the training data is equivalent to finding the symbol pair, whose probability divided by the probabilities of its first symbol followed by its second symbol is the greatest among all symbol pairs. E.g. "u", followed by "g" would have only been merged if the probability of "ug" divided by "u", "g" would have been greater than for any other symbol pair. Intuitively, WordPiece is slightly different to BPE in that it evaluates what it loses by merging two symbols to ensure it’s worth it.

## Unigram

Unigram is a subword tokenization algorithm introduced in Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates (Kudo, 2018). In contrast to BPE or WordPiece, Unigram initializes its base vocabulary to a large number of symbols and progressively trims down each symbol to obtain a smaller vocabulary. The base vocabulary could for instance correspond to all pre-tokenized words and the most common substrings. Unigram is not used directly for any of the models in the transformers, but it’s used in conjunction with SentencePiece.

At each training step, the Unigram algorithm defines a loss (often defined as the log-likelihood) over the training data given the current vocabulary and a unigram language model. Then, for each symbol in the vocabulary, the algorithm computes how much the overall loss would increase if the symbol was to be removed from the vocabulary. Unigram then removes p (with p usually being 10% or 20%) percent of the symbols whose loss increase is the lowest, i.e. those symbols that least affect the overall loss over the training data. This process is repeated until the vocabulary has reached the desired size. The Unigram algorithm always keeps the base characters so that any word can be tokenized.

Because Unigram is not based on merge rules (in contrast to BPE and WordPiece), the algorithm has several ways of tokenizing new text after training. As an example, if a trained Unigram tokenizer exhibits the vocabulary:

`["b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug"]`,

"hugs" could be tokenized both as ["hug", "s"], ["h", "ug", "s"] or ["h", "u", "g", "s"]. So which one to choose? Unigram saves the probability of each token in the training corpus on top of saving the vocabulary so that the probability of each possible tokenization can be computed after training. The algorithm simply picks the most likely tokenization in practice, but also offers the possibility to sample a possible tokenization according to their probabilities.

Those probabilities are defined by the loss the tokenizer is trained on. Assuming that the training data consists of the words

```
L = ∑(i=1..N)log(∑(x∈S(xi)) p(x))
```

## SentencePiece
[SentencePiece Paper](https://arxiv.org/pdf/1808.06226)

### About

All tokenization algorithms described so far have the same problem: It is assumed that the input text uses spaces to separate words. However, not all languages use spaces to separate words. One possible solution is to use language specific pre-tokenizers, e.g. XLM uses a specific Chinese, Japanese, and Thai pre-tokenizer. To solve this problem more generally, SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing (Kudo et al., 2018) treats the input as a raw input stream, thus including the space in the set of characters to use. It then uses the BPE or unigram algorithm to construct the appropriate vocabulary.

The XLNetTokenizer uses SentencePiece for example, which is also why in the example earlier the "▁" character was included in the vocabulary. Decoding with SentencePiece is very easy since all tokens can just be concatenated and "▁" is replaced by a space.

### Paper Summary


It comprises of for main components:

- Normalizer
- Trainer
- Encoder
- Decoder

**Normalizer**: It is a module to normalize semantically equivalent Unicode characters.

**Trainer**: It trains the subword segmentation model from the normalized corpus. We specify the type of subword model as the parameter of Trainer.

**Encoder**: It internally executes Normalizer to normalize the input text and tokenizes it into a subword sequence with the subword model trained by Trainer.

**Decoder**: It converts the subword sequence into the normalized text.

#### Characteristics of the Library

- Lossless Tokenization: Sentence piece implements the Decoder as an inverse operation of the Encoder. This makes sure that all the information to reproduce the normalized text is preserved in the encoder's output. For sake of clarity, SentencePiece first escapes the whitespace with a meta symbol _ (U+2581), and then tokenizes the input into an arbitrary subword sequence, For example "Hello World." -> [Hello] [_wor] [ld] [.]
- Efficient subword training and segmentation: SentencePiece employs several speed-up techniques both for training and segementation to make lossless tokenization with a large amount of raw data. For example given an input sentence (or word) of length *N*,

  - BPE requires *O(N^2)* however SentencePiece takes *O(Nlog(N))*, the merged symbols are managed by a binary heap (priority queue).
  - For unigram language model sentence piece scales linear to the size of the input data.
- Vocabulary id management: The size of the vocab is specified with the `--vocab_size=<size>` flag of `spm_train`. SentencePiece also reserves the following special meta symbols in vocabulary:

  - `<unk>`: Unknown symbol
  - `<s>`: Beginning of a sentence
  - `</s>`: Ending of a sentence
  - `<pad>`: padding
    We can also define custom meta symbols to encode contextual information as virtual tokens.
- Customizable character normalization: By default SentencePiece normalizes the input text with the Unicode NFKC normalization. The normalization rules are specified with the `--normalization_rule_name=nfkc` flag of `spm_train`. The normalization in SentencePiece is implemented with string-to-string mapping and leftmost longest matching. The normalization rules are compiled into a finite state transducer to perform an efficient normalization. SentencePiece supports custom normalization rules defined as a TSV file. When there are ambiguities in the conversion, the longest rule is applied. User defined TSV files are specified with the `--normalization_rule_tsv=<file>` flag of `spm_train`.
    ```txt
    NOTE:
    1. The Original NFKC normalization required CCC (Canonical Combining Class) reordering, which is hard to model in a finite state transducer. SentencePiece does not handle the full CCC reordering and only implements a subset of NFKC normalization.

    2. Note the tabs are used a the delimiter for source and target sequence and spaces are used as the delimiter for individual characters.
    ```

