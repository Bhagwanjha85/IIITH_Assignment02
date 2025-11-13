"""
Generate sample dataset for testing the language model
This creates a simple text corpus for demonstration purposes
"""

import os
from pathlib import Path


def generate_sample_data(output_path='data/train.txt', num_samples=10000):
    """
    Generate sample text data for testing

    Args:
        output_path: Path to save the generated data
        num_samples: Number of sample sentences to generate
    """

    # Templates for generating diverse sentences
    templates = [
        "The {adj} {noun} {verb} {adverb} in the {place}.",
        "A {adj} {noun} is {verb} near the {place}.",
        "The {noun} can {verb} very {adverb}.",
        "I saw a {adj} {noun} {verb} yesterday.",
        "The {adj} {noun} will {verb} tomorrow in the {place}.",
        "Every {noun} should {verb} {adverb}.",
        "The {place} has many {adj} {noun}s.",
        "When the {noun} {verb}s, everyone {verb}s {adverb}.",
        "The most {adj} {noun} {verb}s in the {place}.",
        "A {noun} that {verb}s {adverb} is very {adj}.",
    ]

    # Word lists
    adjectives = [
        'quick', 'lazy', 'happy', 'sad', 'bright', 'dark', 'tall', 'short',
        'big', 'small', 'fast', 'slow', 'beautiful', 'ugly', 'strong', 'weak',
        'smart', 'clever', 'wise', 'foolish', 'brave', 'scared', 'calm', 'angry',
        'gentle', 'rough', 'smooth', 'sharp', 'soft', 'hard'
    ]

    nouns = [
        'cat', 'dog', 'bird', 'fish', 'tree', 'flower', 'mountain', 'river',
        'book', 'pen', 'computer', 'phone', 'car', 'bike', 'house', 'building',
        'student', 'teacher', 'doctor', 'engineer', 'artist', 'musician', 'writer',
        'sun', 'moon', 'star', 'cloud', 'rain', 'snow', 'wind'
    ]

    verbs = [
        'runs', 'walks', 'jumps', 'flies', 'swims', 'sleeps', 'eats', 'drinks',
        'reads', 'writes', 'sings', 'dances', 'plays', 'works', 'studies', 'teaches',
        'builds', 'creates', 'destroys', 'loves', 'hates', 'thinks', 'feels', 'knows',
        'sees', 'hears', 'speaks', 'listens', 'moves', 'stops'
    ]

    adverbs = [
        'quickly', 'slowly', 'carefully', 'carelessly', 'happily', 'sadly',
        'loudly', 'quietly', 'smoothly', 'roughly', 'gently', 'hardly',
        'easily', 'difficultly', 'clearly', 'vaguely', 'well', 'badly',
        'always', 'never', 'sometimes', 'often', 'rarely', 'frequently'
    ]

    places = [
        'park', 'garden', 'forest', 'beach', 'mountain', 'valley', 'city', 'village',
        'school', 'office', 'hospital', 'library', 'museum', 'theater', 'stadium',
        'street', 'road', 'bridge', 'tunnel', 'field', 'farm', 'desert', 'island'
    ]

    # Additional sentence types
    questions = [
        "What is a {adj} {noun}?",
        "Where does the {noun} {verb}?",
        "How can a {noun} {verb} so {adverb}?",
        "Why is the {noun} so {adj}?",
        "When will the {noun} {verb}?",
    ]

    statements = [
        "In the morning, the {noun} {verb}s {adverb}.",
        "During the evening, a {adj} {noun} appears.",
        "At night, the {place} becomes very {adj}.",
        "Throughout the day, many {noun}s {verb}.",
        "Sometimes, a {noun} {verb}s without warning.",
    ]

    # Combine all templates
    all_templates = templates + questions + statements

    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate sentences
    import random
    random.seed(42)

    sentences = []
    for _ in range(num_samples):
        template = random.choice(all_templates)

        sentence = template.format(
            adj=random.choice(adjectives),
            noun=random.choice(nouns),
            verb=random.choice(verbs),
            adverb=random.choice(adverbs),
            place=random.choice(places)
        )

        sentences.append(sentence)

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sentences))

    print(f"Generated {num_samples} sample sentences")
    print(f"Saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")

    # Print sample
    print("\nSample sentences:")
    for i in range(5):
        print(f"  {i+1}. {sentences[i]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate sample training data')
    parser.add_argument('--output', type=str, default='data/train.txt',
                       help='Output file path')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of sentences to generate')

    args = parser.parse_args()

    generate_sample_data(args.output, args.num_samples)

    print("\n✓ Sample data generation completed!")
    print("\nNext steps:")
    print("1. Review the generated data in data/train.txt")
    print("2. Run: python train.py --scenario all --epochs 20")
    print("3. Check outputs in outputs/ folder")