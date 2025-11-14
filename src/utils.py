import torch
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
import numpy as np


class Vocabulary:

    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()

        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.SOS_TOKEN = '<SOS>'  
        self.EOS_TOKEN = '<EOS>' 

      
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.SOS_TOKEN: 2,
            self.EOS_TOKEN: 3
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def build_vocab(self, texts):
        # It Count word's frequencies
        for text in texts:
            words = self.tokenize(text)
            self.word_freq.update(words)

        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        print(f"Vocabulary built: {len(self.word2idx)} tokens")
        print(f"Min frequency threshold: {self.min_freq}")

    def tokenize(self, text):
        
        # Convert into the lowercase
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\.\,\!\?\'\-]", "", text)
        tokens = text.split()
        return tokens

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.word2idx.get(token, self.word2idx[self.UNK_TOKEN])
                for token in tokens]

    def decode(self, indices):
        words = [self.idx2word.get(idx, self.UNK_TOKEN) for idx in indices]
        return ' '.join(words)

    def __len__(self):
        return len(self.word2idx)


class TextDataset(Dataset):

    def __init__(self, texts, vocab, seq_len=35):
        self.vocab = vocab
        self.seq_len = seq_len

        # Encode all texts and concatenate
        self.data = []
        for text in texts:
            encoded = vocab.encode(text)
            self.data.extend(encoded)

        self.data = torch.tensor(self.data, dtype=torch.long)
        print(f"Dataset created: {len(self.data)} tokens, {len(self)} sequences")

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
       
        input_seq = self.data[idx:idx + self.seq_len]

        target_seq = self.data[idx + 1:idx + self.seq_len + 1]

        return input_seq, target_seq


def load_data(file_path):
    """
    Load text data from file

    Args:
        file_path: Path to text file

    Returns:
        List of text lines
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()

    # Remove empty lines 
    texts = [text.strip() for text in texts if text.strip()]

    print(f"Loaded {len(texts)} lines from {file_path}")
    return texts


def split_data(texts, train_ratio=0.8, val_ratio=0.1):
    """
    Split data into train, validation, and test sets

    Args:
        texts: List of text strings
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set

    Returns:
        train_texts, val_texts, test_texts
    """
    n = len(texts)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_texts = texts[:train_size]
    val_texts = texts[train_size:train_size + val_size]
    test_texts = texts[train_size + val_size:]

    print(f"\nData split:")
    print(f"  Train: {len(train_texts)} samples")
    print(f"  Val:   {len(val_texts)} samples")
    print(f"  Test:  {len(test_texts)} samples")

    return train_texts, val_texts, test_texts


def create_dataloaders(train_texts, val_texts, test_texts, vocab,
                       seq_len=35, batch_size=32):
    """
    Create PyTorch DataLoaders for train, validation, and test sets

    Args:
        train_texts, val_texts, test_texts: Text data
        vocab: Vocabulary object
        seq_len: Sequence length
        batch_size: Batch size

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = TextDataset(train_texts, vocab, seq_len)
    val_dataset = TextDataset(val_texts, vocab, seq_len)
    test_dataset = TextDataset(test_texts, vocab, seq_len)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )

    return train_loader, val_loader, test_loader


def set_seed(seed=42):
    """
    Set random seeds for reproducibility

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility")


if __name__ == "__main__":
    # Test the utilities
    print("Testing data processing utilities\n")

    # Test vocabulary
    sample_texts = [
        "Hello world! This is a test.",
        "Testing the vocabulary builder.",
        "Hello again! This is another test."
    ]

    vocab = Vocabulary(min_freq=1)
    vocab.build_vocab(sample_texts)

    print(f"\nVocabulary size: {len(vocab)}")

    # Test encoding/decoding
    text = "Hello world"
    encoded = vocab.encode(text)
    decoded = vocab.decode(encoded)
    print(f"\nOriginal: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    # Test dataset
    dataset = TextDataset(sample_texts, vocab, seq_len=5)
    print(f"\nDataset length: {len(dataset)}")

    if len(dataset) > 0:
        input_seq, target_seq = dataset[0]
        print(f"Sample input: {input_seq}")
        print(f"Sample target: {target_seq}")
        print(f"Decoded input: {vocab.decode(input_seq.tolist())}")
        print(f"Decoded target: {vocab.decode(target_seq.tolist())}")


    print("\n Utilities test completed successfully!")
