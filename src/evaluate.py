"""
Evaluation script with text generation for Pride and Prejudice model
"""

import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
import numpy as np
import os
import sys

# Add project root to path to ensure imports from model and utils work
# This assumes model.py and utils.py are in the parent directory of where main is called
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from model import create_model
from utils import Vocabulary


def load_checkpoint_and_vocab(checkpoint_path, vocab_path, device='cuda'):
    """Load trained model and vocabulary"""

    # Load vocabulary
    vocab = Vocabulary()
    vocab.load(vocab_path)

    # Load checkpoint
    # Handling NumPy globals for PyTorch loading compatibility
    import numpy
    import torch.serialization
    torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = create_model(checkpoint['vocab_size'], checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Val Perplexity: {checkpoint['val_perplexity']:.2f}")
    print(f"✓ Vocabulary loaded: {len(vocab)} tokens")

    return model, vocab, checkpoint


def generate_text(model, vocab, start_text, max_length=100, temperature=0.8,
                  top_k=50, device='cuda'):
    """
    Generate text in the style of Pride and Prejudice

    Args:
        model: Trained language model
        vocab: Vocabulary object
        start_text: Starting prompt
        max_length: Maximum words to generate
        temperature: Sampling temperature (0.7-1.0 for coherent text)
        top_k: Top-k sampling (50 is good for creative text)
        device: Device (CPU or CUDA)

    Returns:
        Generated text
    """
    model.eval()

    # Encode starting text
    tokens = vocab.encode(start_text)
    if len(tokens) == 0:
        tokens = [vocab.word2idx[vocab.SOS_TOKEN]]

    # Prepare initial input sequence
    input_seq = torch.tensor([tokens], dtype=torch.long).to(device)
    generated_tokens = tokens.copy()

    # Initialize hidden state
    # We pass None initially, and the model handles the first iteration's hidden state.
    hidden = None

    # Use the last token of the prompt as the input for the *next* prediction iteration
    # To start generation, the input_seq for the model only needs the last token to predict the next one
    if len(tokens) > 1:
        # Use the whole prompt for the first forward pass to prime the hidden state
        # input_seq is already the full prompt
        with torch.no_grad():
            output, hidden = model(input_seq)
            # The next input will be the predicted token, so reset input_seq to just the last token
            # This is crucial for generation loop efficiency
            input_seq = input_seq[:, -1].unsqueeze(0)
    elif len(tokens) == 1:
        # If the prompt was just one token, input_seq is fine, hidden is None
        pass


    with torch.no_grad():

        for _ in range(max_length):
            # Forward pass: input_seq shape is (1, 1) or (1, prompt_len) initially
            output, hidden = model(input_seq, hidden)

            # Get the predictions for the *last* word in the output sequence
            # output shape: (batch_size, seq_len, vocab_size)
            logits = output[0, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
                logits_filtered = torch.full_like(logits, -float('Inf'))
                logits_filtered.scatter_(0, indices, values)
                logits = logits_filtered

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Add to generated tokens
            generated_tokens.append(next_token)

            # Update input for next iteration: input_seq must now be (1, 1) containing only the last generated token
            input_seq = torch.tensor([[next_token]], dtype=torch.long).to(device)

            # Stop if we generate EOS token or special token
            if next_token == vocab.word2idx.get(vocab.EOS_TOKEN, -1) or \
               next_token == vocab.word2idx.get(vocab.PAD_TOKEN, -1):
                # Only include tokens up to but not including the special token
                generated_tokens = generated_tokens[:-1]
                break

    # Decode tokens to text
    generated_text = vocab.decode(generated_tokens)

    # Clean up text for better formatting
    generated_text = generated_text.replace(' .', '.').replace(' ,', ',')
    generated_text = generated_text.replace(' !', '!').replace(' ?', '?')
    generated_text = generated_text.replace(' ;', ';').replace(' :', ':')
    generated_text = generated_text.replace(" 's", "'s").replace(" 't", "'t")
    generated_text = generated_text.replace(" 're", "'re").replace(" 'll", "'ll")
    generated_text = generated_text.replace(" 'm", "'m").replace(" 've", "'ve")

    return generated_text


def interactive_generation(model, vocab, device='cuda'):
    """Interactive text generation"""

    print("\n" + "="*80)
    print(" PRIDE AND PREJUDICE - INTERACTIVE TEXT GENERATION")
    print("="*80)
    print("\nGenerate text in the style of Jane Austen!")
    print("Type your prompt and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    # Sample prompts
    sample_prompts = [
        "It is a truth universally acknowledged",
        "Mr. Darcy",
        "Elizabeth",
        "The ball at Netherfield",
        "Mrs. Bennet"
    ]

    print("Sample prompts you can try:")
    for i, prompt in enumerate(sample_prompts, 1):
        print(f"  {i}. {prompt}")
    print()

    while True:
        try:
            prompt = input(" Enter prompt (or number 1-5 for sample): ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\n Goodbye!")
                break

            # Handle sample prompt selection
            if prompt.isdigit() and 1 <= int(prompt) <= len(sample_prompts):
                prompt = sample_prompts[int(prompt) - 1]
                print(f"Using prompt: \"{prompt}\"")

            if not prompt:
                print(" Please enter a non-empty prompt.\n")
                continue

            print(f"\n Generating text...")

            # Generate with different settings for variety
            generated = generate_text(
                model, vocab, prompt,
                max_length=100,
                temperature=0.8,
                top_k=50,
                device=device
            )

            print(f"\n{'─'*80}")
            print(f"Generated text:\n")
            print(f"{generated}")
            print(f"{'─'*80}\n")

        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f" Error: {e}\n")


def batch_generate_samples(model, vocab, device='cuda', num_samples=5):
    """Generate multiple samples with different prompts"""

    prompts = [
        "It is a truth universally acknowledged",
        "Mr. Darcy was",
        "Elizabeth Bennet",
        "The ladies of Longbourn",
        "At the ball"
    ]

    print("\n" + "="*80)
    print(" GENERATING SAMPLE TEXTS IN JANE AUSTEN'S STYLE")
    print("="*80)

    for i, prompt in enumerate(prompts[:num_samples], 1):
        print(f"\n{'─'*80}")
        print(f"Sample {i}: Prompt = \"{prompt}\"")
        print(f"{'─'*80}")

        generated = generate_text(
            model, vocab, prompt,
            max_length=80,
            temperature=0.8,
            top_k=50,
            device=device
        )

        print(f"\n{generated}\n")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Pride and Prejudice Language Model')
    parser.add_argument('--checkpoint', type=str,
                       default='outputs/checkpoints/best_model_best_fit.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str,
                       default='outputs/vocab.pkl',
                       help='Path to vocabulary file')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['generate', 'interactive', 'samples'],
                       help='Evaluation mode')
    parser.add_argument('--prompt', type=str,
                       default='It is a truth universally acknowledged',
                       help='Starting text for generation')
    parser.add_argument('--max_length', type=int, default=100,
                       help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (0.7-1.0 for Austen style)')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print(" CUDA not available, using CPU")
        args.device = 'cpu'

    print(f" Using device: {args.device}")

    # Load model and vocabulary
    try:
        print("\n Loading model and vocabulary...")
        model, vocab, checkpoint = load_checkpoint_and_vocab(
            args.checkpoint, args.vocab, args.device
        )
    except Exception as e:
        print(f"\n Error loading model or vocabulary. Please ensure training has run and files exist:")
        print(f" Checkpoint Path: {args.checkpoint}")
        print(f" Vocab Path: {args.vocab}")
        print(f" Details: {e}")
        return

    # Run based on mode
    if args.mode == 'generate':
        print(f"\n Generating text with prompt: \"{args.prompt}\"")
        print("─" * 80)

        generated = generate_text(
            model, vocab, args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device
        )

        print(f"\n Generated text:\n")
        print(f"{generated}\n")
        print("─" * 80)

    elif args.mode == 'samples':
        batch_generate_samples(model, vocab, args.device)

    elif args.mode == 'interactive':
        interactive_generation(model, vocab, args.device)


if __name__ == "__main__":
    main()