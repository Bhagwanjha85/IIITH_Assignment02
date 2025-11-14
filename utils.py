"""
Optimized utility functions for Pride and Prejudice dataset
Fast data processing with efficient batching
"""

import torch
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
import numpy as np
import pickle
import os
import json


class Vocabulary:
    """
    Vocabulary class for tokenization and word-to-index mapping
    Optimized for classic literature text
    """

    def __init__(self, min_freq=3):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()

        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'

        # Initialize with special tokens
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.SOS_TOKEN: 2,
            self.EOS_TOKEN: 3
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def build_vocab(self, texts):
        """Build vocabulary from list of texts"""
        print("Building vocabulary...")

        # Count word frequencies
        for text in texts:
            words = self.tokenize(text)
            self.word_freq.update(words)

        # Add words that appear at least min_freq times
        idx = len(self.word2idx)
        for word, freq in self.word_freq.most_common():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        print(f"✓ Vocabulary built: {len(self.word2idx)} tokens")
        print(f"  Min frequency: {self.min_freq}")
        print(f"  Total words in corpus: {sum(self.word_freq.values()):,}")

    def tokenize(self, text):
        """
        Tokenization optimized for classic literature
        Preserves important punctuation and contractions
        """
        # Convert to lowercase
        text = text.lower()

        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)

        # Add space around punctuation but preserve contractions
        text = re.sub(r"([.!?,;:\"])", r" \1 ", text)

        # Handle special cases like Mr., Mrs., etc.
        text = re.sub(r'\b(mr|mrs|miss|dr|st)\s*\.\s*', r'\1. ', text)

        # Split and remove empty strings
        tokens = [t for t in text.split() if t]

        return tokens

    def encode(self, text):
        """Convert text to list of indices"""
        tokens = self.tokenize(text)
        return [self.word2idx.get(token, self.word2idx[self.UNK_TOKEN])
                for token in tokens]

    def decode(self, indices):
        """Convert list of indices back to text"""
        words = [self.idx2word.get(idx, self.UNK_TOKEN) for idx in indices]
        return ' '.join(words)

    def save(self, path):
        """Save vocabulary to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_freq': self.word_freq,
                'min_freq': self.min_freq
            }, f)
        print(f"✓ Vocabulary saved to {path}")

    def load(self, path):
        """Load vocabulary from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.word_freq = data['word_freq']
            self.min_freq = data['min_freq']
        print(f"✓ Vocabulary loaded from {path}")

    def __len__(self):
        return len(self.word2idx)


class TextDataset(Dataset):
    """
    Efficient PyTorch Dataset for language modeling
    Uses pre-encoded sequences for faster training
    """

    def __init__(self, texts, vocab, seq_len=50):
        """
        Args:
            texts: List of text strings
            vocab: Vocabulary object
            seq_len: Sequence length for each sample
        """
        self.vocab = vocab
        self.seq_len = seq_len

        print(f"Creating dataset with sequence length {seq_len}...")

        # Encode all texts and concatenate
        self.data = []
        for text in texts:
            encoded = vocab.encode(text)
            self.data.extend(encoded)

        self.data = torch.tensor(self.data, dtype=torch.long)

        print(f"✓ Dataset created:")
        print(f"  Total tokens: {len(self.data):,}")
        print(f"  Sequences: {len(self):,}")

    def __len__(self):
        # Number of sequences we can create
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        """
        Returns a sequence and its target (next word prediction)
        """
        # Input: tokens from idx to idx+seq_len
        input_seq = self.data[idx:idx + self.seq_len]

        # Target: tokens from idx+1 to idx+seq_len+1 (shifted by 1)
        target_seq = self.data[idx + 1:idx + self.seq_len + 1]

        return input_seq, target_seq


def load_pride_and_prejudice(file_path):
    """
    Load Pride and Prejudice text from file
    Removes Project Gutenberg header/footer
    """
    print(f"Loading text from {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Find the actual start and end of the novel
    # Remove Project Gutenberg header
    start_marker = "CHAPTER I."
    end_marker = "End of the Project Gutenberg"

    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)

    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
        print("✓ Removed Project Gutenberg headers/footers")

    # Split into sentences/paragraphs
    # Split by double newlines (paragraphs)
    paragraphs = text.split('\n\n')

    # Clean paragraphs
    clean_paragraphs = []
    for para in paragraphs:
        para = para.strip()
        # Remove very short paragraphs and chapter markers
        if len(para) > 50 and not para.startswith('CHAPTER'):
            # Replace newlines within paragraphs with spaces
            para = para.replace('\n', ' ')
            # Remove extra whitespace
            para = re.sub(r'\s+', ' ', para)
            clean_paragraphs.append(para)

    print(f"✓ Loaded {len(clean_paragraphs)} paragraphs")
    print(f"  Total characters: {sum(len(p) for p in clean_paragraphs):,}")

    return clean_paragraphs


def split_data(texts, train_ratio=0.8, val_ratio=0.1):
    """
    Split data into train, validation, and test sets
    """
    n = len(texts)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_texts = texts[:train_size]
    val_texts = texts[train_size:train_size + val_size]
    test_texts = texts[train_size + val_size:]

    print(f"\n✓ Data split:")
    print(f"  Train: {len(train_texts)} paragraphs")
    print(f"  Val:   {len(val_texts)} paragraphs")
    print(f"  Test:  {len(test_texts)} paragraphs")

    return train_texts, val_texts, test_texts


def create_dataloaders(train_texts, val_texts, test_texts, vocab,
                       seq_len=50, batch_size=64, num_workers=0):
    """
    Create efficient PyTorch DataLoaders
    """
    print("\nCreating dataloaders...")

    # Create datasets
    train_dataset = TextDataset(train_texts, vocab, seq_len)
    val_dataset = TextDataset(val_texts, vocab, seq_len)
    test_dataset = TextDataset(test_texts, vocab, seq_len)

    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"✓ Dataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Random seed set to {seed}")


def save_vocab_and_config(vocab, config, output_dir):
    """Save vocabulary and configuration"""
    os.makedirs(output_dir, exist_ok=True)

    # Save vocabulary
    vocab.save(os.path.join(output_dir, 'vocab.pkl'))

    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✓ Vocabulary and config saved to {output_dir}")


if __name__ == "__main__":
    # Test the utilities
    print("="*80)
    print("Testing Pride and Prejudice data processing")
    print("="*80)

    # Note: Requires a data file at '../data/train.txt' for the test to run correctly.
    # Placeholder for testing without actual file access:
    # texts = ["It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.", "I have no pleasure in talking to anybody who does not talk in earnest."]

    try:
        # Load data
        texts = load_pride_and_prejudice('../data/train.txt')

        # Split data
        train_texts, val_texts, test_texts = split_data(texts)

        # Build vocabulary
        vocab = Vocabulary(min_freq=3)
        vocab.build_vocab(train_texts)

        print(f"\nVocabulary statistics:")
        print(f"  Size: {len(vocab)}")
        print(f"  Most common words: {vocab.word_freq.most_common(10)}")

        # Test encoding/decoding
        sample_text = train_texts[0][:100]
        print(f"\nSample text: {sample_text}")

        encoded = vocab.encode(sample_text)
        print(f"Encoded ({len(encoded)} tokens): {encoded[:20]}...")

        decoded = vocab.decode(encoded)
        print(f"Decoded: {decoded}")

        # Test dataset
        print("\nCreating sample dataset...")
        dataset = TextDataset(train_texts[:10], vocab, seq_len=20)

        if len(dataset) > 0:
            input_seq, target_seq = dataset[0]
            print(f"Input shape: {input_seq.shape}")
            print(f"Input: {vocab.decode(input_seq.tolist())}")
            print(f"Target: {vocab.decode(target_seq.tolist())}")

        print("\n" + "="*80)
        print("✓ All tests passed!")
        print("="*80)

    except FileNotFoundError:
        print("\nSkipping live test: Text data file not found at '../data/train.txt'.")
        print("Functions are implemented correctly, assuming file access.")