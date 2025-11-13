"""
Neural Language Model Implementation using LSTM
Assignment 2: Language Model Training from Scratch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMLanguageModel(nn.Module):
    """
    LSTM-based Language Model for next-word prediction

    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of LSTM hidden state
        num_layers: Number of LSTM layers
        dropout: Dropout probability (default: 0.3)
        tie_weights: Whether to tie input and output embeddings (default: False)
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 dropout=0.3, tie_weights=False):
        super(LSTMLanguageModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Embedding layer: converts word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layers: process sequences and capture long-term dependencies
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)

        # Output layer: projects hidden state to vocabulary size
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Weight tying: shares weights between embedding and output layers
        # This reduces parameters and often improves performance
        if tie_weights:
            if hidden_dim != embedding_dim:
                raise ValueError("When using tied weights, hidden_dim must equal embedding_dim")
            self.fc.weight = self.embedding.weight

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize model weights with uniform distribution"""
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)

    def forward(self, x, hidden=None):
        """
        Forward pass through the model

        Args:
            x: Input tensor of shape (batch_size, seq_len)
            hidden: Previous hidden state (optional)

        Returns:
            output: Predictions of shape (batch_size, seq_len, vocab_size)
            hidden: Updated hidden state
        """
        # Get embeddings: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        embedded = self.dropout_layer(embedded)

        # Pass through LSTM
        if hidden is not None:
            lstm_out, hidden = self.lstm(embedded, hidden)
        else:
            lstm_out, hidden = self.lstm(embedded)

        # Apply dropout
        lstm_out = self.dropout_layer(lstm_out)

        # Project to vocabulary size: (batch_size, seq_len, vocab_size)
        output = self.fc(lstm_out)

        return output, hidden

    def init_hidden(self, batch_size, device):
        """
        Initialize hidden state for LSTM

        Args:
            batch_size: Batch size
            device: Device (CPU or CUDA)

        Returns:
            Tuple of (hidden_state, cell_state)
        """
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_dim).to(device))


def create_model(vocab_size, config):
    """
    Factory function to create model with specific configuration

    Args:
        vocab_size: Size of vocabulary
        config: Dictionary with model hyperparameters

    Returns:
        LSTMLanguageModel instance
    """
    return LSTMLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        tie_weights=config.get('tie_weights', False)
    )


def count_parameters(model):
    """Count total trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Model Configurations for Different Scenarios
CONFIGS = {
    'underfit': {
        'embedding_dim': 128,
        'hidden_dim': 128,
        'num_layers': 1,
        'dropout': 0.0,
        'tie_weights': False,
        'description': 'Small model to demonstrate underfitting'
    },
    'overfit': {
        'embedding_dim': 512,
        'hidden_dim': 512,
        'num_layers': 3,
        'dropout': 0.0,  # No dropout = prone to overfitting
        'tie_weights': False,
        'description': 'Large model with no regularization to demonstrate overfitting'
    },
    'best_fit': {
        'embedding_dim': 256,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'tie_weights': True,
        'description': 'Optimal model with proper regularization'
    }
}


if __name__ == "__main__":
    # Test the model
    print("Testing LSTM Language Model Implementation\n")

    vocab_size = 10000
    batch_size = 32
    seq_len = 20

    for scenario, config in CONFIGS.items():
        print(f"--- {scenario.upper()} Scenario ---")
        print(f"Description: {config['description']}")

        model = create_model(vocab_size, config)
        param_count = count_parameters(model)

        print(f"Model Parameters: {param_count:,}")
        print(f"Config: {config}\n")

        # Test forward pass
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
        output, hidden = model(dummy_input)

        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Hidden state shape: {hidden[0].shape}\n")

    print("✓ Model implementation test completed successfully!")