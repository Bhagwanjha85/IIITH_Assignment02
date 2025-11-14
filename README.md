# Neural Language Model - PyTorch Implementation

A complete implementation of an LSTM-based Neural Language Model from scratch using PyTorch. This project demonstrates understanding of sequence models, training dynamics, and the concepts of underfitting, overfitting, and optimal model selection.

## Project Overview

This project implements a character-level language model that learns to predict the next word in a sequence. The model is trained and evaluated under three different scenarios:

1. **Underfitting**: Small model with insufficient capacity
2. **Overfitting**: Large model with no regularization
3. **Best Fit**: Optimally configured model with proper regularization

## Prerequisites
Python 3.8 or higher  
• CUDA-capable GPU (optional, but recommended)  
• 8GB+ RAM  
• ~2GB free disk space 

## Quick open in local system

### 1. Clone the Repository

```bash
git clone https://github.com/Bhagwanjha85/IIITH_Assignment02.git
cd src
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install torch numpy matplotlib tensorboard tqdm
```

### 3. Prepare Dataset( Dataset is already given by IIITH)

we used my own dataset `data/train.txt`. 

### 4. Train the Model

#### Train all the three scenarios:
```bash
python train.py --scenario all --epochs 20
```

#### Train a specific scenario:
```bash
# Best fit model 
python train.py --scenario best_fit --epochs 20

# Underfit model
python train.py --scenario underfit --epochs 20

# Overfit model
python train.py --scenario overfit --epochs 20
```

#### When we Training on GPU (Google Colab / Kaggle):
```bash
python train.py --scenario all --device cuda --epochs 30
```

### 5. Evaluate the Model

#### Generate text:
```bash
python evaluate.py --checkpoint outputs/checkpoints/best_model_best_fit.pt \
                   --mode generate \
                   --prompt "It is a truth universally acknowledged" \
                   --max_length 50
```

#### Calculate perplexity:
```bash
python evaluate.py --checkpoint outputs/checkpoints/best_model_best_fit.pt \
                   --mode perplexity \
                   --data_path data/train.txt
```

#### Interactive generation:
```bash
python evaluate.py --checkpoint outputs/checkpoints/best_model_best_fit.pt \
                   --mode interactive
```

## Model Configurations

### Underfitting Scenario
```python
{
    'embedding_dim': 128,
    'hidden_dim': 128,
    'num_layers': 1,
    'dropout': 0.0,
    'tie_weights': False
}
```
- **Purpose**: Show a model with insufficient capacity
- **Expected**: High training and validation loss
- **Parameters**: ~500K

### Overfitting Scenario
```python
{
    'embedding_dim': 512,
    'hidden_dim': 512,
    'num_layers': 3,
    'dropout': 0.0,  # No regularization!
    'tie_weights': False
}
```
- **Purpose**: Demonstrates overfitting without regularization
- **Expected**: Low training loss, high validation loss
- **Parameters**: ~8M

### Best Fit Scenario
```python
{
    'embedding_dim': 256,
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.3,
    'tie_weights': True
}
```
- **Purpose**: Optimal balance between capacity and generalization
- **Expected**: Balanced training and validation loss
- **Parameters**: ~2M

## Expected Results

After training, we will find:

1. **Loss Curves** in `outputs/plots/`:
   - `loss_curve_underfit.png`
   - `loss_curve_overfit.png`
   - `loss_curve_best_fit.png`

2. **Model Checkpoints** in `outputs/checkpoints/`:
   - `best_model_underfit.pt`
   - `best_model_overfit.pt`
   - `best_model_best_fit.pt`

3. **Results Summary** in `outputs/`:
   - `results_underfit.txt`
   - `results_overfit.txt`
   - `results_best_fit.txt`
     



### Loss Curves Interpretation

**Underfitting:**
```
Training Loss: High and plateaus
Validation Loss: High and plateaus
Gap: Small
```

**Overfitting:**
```
Training Loss: Continues decreasing
Validation Loss: Increases after initial decrease
Gap: Large and growing
```

**Best Fit:**
```
Training Loss: Steady decrease
Validation Loss: Follows training loss closely
Gap: Small and stable
```

## Key Observations

### Why Underfitting Happens
- Model is too simple (few parameters)
- Cannot capture patterns in data
- High bias, low variance

### Why Overfitting Happens
- Model is too complex
- No regularization (dropout)
- Memorizes training data
- Low bias, high variance

## Reproducibility

All experiments use fixed random seeds:
```python
set_seed(42)  # Ensures reproducible results
```

To reproduce results:
1. Use the same dataset
2. Run with default hyperparameters
3. Use the same PyTorch version

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)
- 4GB+ RAM recommended
- GPU with 4GB+ VRAM (for large models)

## Running on Google Colab

```python
# Install dependencies
!pip install torch torchvision

# Clone repository
!git clone https://github.com/yourusername/neural-language-model.git
%cd neural-language-model

# Upload your dataset to Colab
from google.colab import files
uploaded = files.upload()  # Upload train.txt

# Move to data folder
!mkdir -p data
!mv train.txt data/

# Train
!python train.py --scenario all --device cuda --epochs 20

# Download results
files.download('outputs/plots/loss_curve_best_fit.png')
```

## Visualization

The training script automatically generates publication-quality plots showing:
- Training vs Validation Loss
- Clear separation between scenarios
- Epoch-wise progression

## Assignment Deliverables Checklist

-  Complete the PyTorch implementation from scratch
-  Training script with three scenarios 
-  Loss curves (underfit, overfit, best fit)
-  Perplexity metrics on test set
-  Model checkpoints
-  detailed README to showcase our project
-  Reproducible code with fixed seeds
-  Text generation capability

