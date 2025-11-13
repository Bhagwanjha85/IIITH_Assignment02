"""
Training script for Neural Language Model
Supports three scenarios: underfit, overfit, and best_fit
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import time
from pathlib import Path

from model import create_model, CONFIGS, count_parameters
from utils import (Vocabulary, load_data, split_data, create_dataloaders,
                   set_seed)


def train_epoch(model, train_loader, criterion, optimizer, device, clip_grad=5.0):
    """
    Train for one epoch

    Args:
        model: Language model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (CPU or CUDA)
        clip_grad: Gradient clipping value

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output, _ = model(input_seq)

        # Reshape for loss calculation
        # output: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
        # target: (batch_size, seq_len) -> (batch_size * seq_len)
        output = output.view(-1, output.size(-1))
        target = target_seq.view(-1)

        # Calculate loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        # Update weights
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model on validation/test set

    Args:
        model: Language model
        data_loader: Data loader
        criterion: Loss function
        device: Device (CPU or CUDA)

    Returns:
        Average loss and perplexity
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for input_seq, target_seq in data_loader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            # Forward pass
            output, _ = model(input_seq)

            # Reshape for loss calculation
            output = output.view(-1, output.size(-1))
            target = target_seq.view(-1)

            # Calculate loss
            loss = criterion(output, target)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = np.exp(avg_loss)

    return avg_loss, perplexity


def plot_losses(train_losses, val_losses, scenario, save_path):
    """
    Plot training and validation losses

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        scenario: Scenario name (underfit/overfit/best_fit)
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training vs Validation Loss - {scenario.upper()}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def train_model(scenario, data_path, output_dir, epochs=20, batch_size=32,
                seq_len=35, learning_rate=0.0001, device='cuda'):
    """
    Main training function

    Args:
        scenario: Scenario name (underfit/overfit/best_fit)
        data_path: Path to dataset
        output_dir: Directory to save outputs
        epochs: Number of training epochs
        batch_size: Batch size
        seq_len: Sequence length
        learning_rate: Learning rate
        device: Device (CPU or CUDA)
    """
    print(f"\n{'='*80}")
    print(f"Training Scenario: {scenario.upper()}")
    print(f"{'='*80}\n")

    # Set random seed for reproducibility
    set_seed(42)

    # Create output directories
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    checkpoints_dir = output_dir / 'checkpoints'
    plots_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    print("Loading data...")
    texts = load_data(data_path)
    train_texts, val_texts, test_texts = split_data(texts)

    # Build vocabulary
    print("\nBuilding vocabulary...")
    vocab = Vocabulary(min_freq=2)
    vocab.build_vocab(train_texts)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_texts, val_texts, test_texts, vocab, seq_len, batch_size
    )

    # Create model
    print(f"\nCreating model with configuration: {scenario}")
    config = CONFIGS[scenario]
    model = create_model(len(vocab), config)
    model = model.to(device)

    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")
    print(f"Model configuration:\n{config}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # Training loop
    print(f"\n{'='*80}")
    print("Starting Training")
    print(f"{'='*80}\n")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate
        val_loss, val_perplexity = evaluate(model, val_loader, criterion, device)

        # Save losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step(val_loss)

        # Print progress
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Time: {elapsed:5.1f}s | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Perplexity: {val_perplexity:.2f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoints_dir / f'best_model_{scenario}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_perplexity': val_perplexity,
                'config': config,
                'vocab_size': len(vocab)
            }, checkpoint_path)
            print(f"  ✓ Best model saved (Val Loss: {val_loss:.4f})")

    # Final evaluation on test set
    print(f"\n{'='*80}")
    print("Final Evaluation on Test Set")
    print(f"{'='*80}\n")

    test_loss, test_perplexity = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Perplexity: {test_perplexity:.2f}")

    # Plot losses
    plot_path = plots_dir / f'loss_curve_{scenario}.png'
    plot_losses(train_losses, val_losses, scenario, plot_path)

    # Save final results
    results = {
        'scenario': scenario,
        'config': config,
        'train_loss_final': train_losses[-1],
        'val_loss_final': val_losses[-1],
        'val_loss_best': best_val_loss,
        'test_loss': test_loss,
        'test_perplexity': test_perplexity,
        'num_parameters': param_count,
        'epochs_trained': epochs
    }

    results_path = output_dir / f'results_{scenario}.txt'
    with open(results_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    print(f"\nResults saved to {results_path}")
    print(f"\n{'='*80}")
    print(f"Training completed for {scenario.upper()}!")
    print(f"{'='*80}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train Neural Language Model')
    parser.add_argument('--scenario', type=str, default='best_fit',
                       choices=['underfit', 'overfit', 'best_fit', 'all'],
                       help='Training scenario')
    parser.add_argument('--data_path', type=str, default='data/train.txt',
                       help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--seq_len', type=int, default=35,
                       help='Sequence length')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'

    print(f"Using device: {args.device}")

    # Train based on scenario
    if args.scenario == 'all':
        scenarios = ['underfit', 'overfit', 'best_fit']
    else:
        scenarios = [args.scenario]

    all_results = {}
    for scenario in scenarios:
        results = train_model(
            scenario=scenario,
            data_path=args.data_path,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            learning_rate=args.lr,
            device=args.device
        )
        all_results[scenario] = results

    # Print summary
    if len(scenarios) > 1:
        print(f"\n{'='*80}")
        print("SUMMARY OF ALL SCENARIOS")
        print(f"{'='*80}\n")

        for scenario, results in all_results.items():
            print(f"{scenario.upper()}:")
            print(f"  Parameters: {results['num_parameters']:,}")
            print(f"  Best Val Loss: {results['val_loss_best']:.4f}")
            print(f"  Test Perplexity: {results['test_perplexity']:.2f}")
            print()


if __name__ == "__main__":
    main()