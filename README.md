TabICL: A PyTorch Lightning Implementation
This repository contains a PyTorch Lightning implementation of the TabICL model, as described in the paper "TabICL: A Tabular Foundation Model for In-Context Learning on Large Data" by Qu et al.

The implementation is self-contained within tabicl_lightning.py for ease of use.

Model Architecture
The model is composed of three main transformer-based stages, following the architecture outlined in the paper:

Distribution-aware Column-wise Embedding (TF_col): Implemented as SetTransformer, this module processes each feature column independently. It uses Induced Self-Attention Blocks (ISAB) to efficiently capture the statistical properties of each column and create powerful feature embeddings.

Context-aware Row-wise Interaction (TF_row): The RowInteractionTransformer takes the feature embeddings for each row, prepends learnable [CLS] tokens, and uses a standard transformer encoder to model inter-feature dependencies. The paper proposes using Rotary Positional Embeddings (RoPE) to distinguish between identically distributed features; a placeholder for this is included. The final row embedding is formed by concatenating the outputs of the four [CLS] tokens.

Dataset-wise In-Context Learning (TF_icl): The ICLTransformer performs the final prediction step. It takes the embedded training rows (combined with their label embeddings) and the embedded test rows as a single sequence. Using a causal attention mask, it predicts the classes for all test samples in a single forward pass, leveraging the training samples as in-context examples.

File Structure
tabicl_lightning.py: Contains the complete source code, including:

Helper modules (RotaryPositionalEmbedding, MAB, ISAB).

The three core components (SetTransformer, RowInteractionTransformer, ICLTransformer).

The main TabICL LightningModule which orchestrates the components and defines the training logic.

A simple if __name__ == '__main__': block demonstrating how to instantiate the model and run a forward pass with dummy data.

Requirements
Python 3.x

PyTorch

PyTorch Lightning

You can install the required libraries via pip:

pip install torch pytorch-lightning

Usage
1. Model Initialization
First, define a configuration dictionary and instantiate the TabICL model.

from tabicl_lightning import TabICL

# --- Configuration ---
config = {
    "d_model": 128,
    "num_classes": 10,
    "learning_rate": 1e-4,
    # TF_col
    "tf_col_heads": 4,
    "tf_col_inds": 32, # Number of inducing points
    "tf_col_blocks": 2,
    # TF_row
    "tf_row_layers": 3,
    "tf_row_heads": 8,
    "tf_row_ff": 512,
    # TF_icl
    "tf_icl_layers": 12,
    "tf_icl_heads": 4,
    "tf_icl_ff": 1024
}

model = TabICL(config)
print(model)

2. Data Preparation
The model expects data in a specific format for its forward pass, representing a batch of in-context learning tasks.

x_train: A tensor of shape (batch_size, n_train_samples, n_features) containing the training features for the context.

y_train: A tensor of shape (batch_size, n_train_samples) containing the training labels for the context.

x_test: A tensor of shape (batch_size, n_test_samples, n_features) containing the test features to be predicted.

import torch

# --- Dummy Data for demonstration ---
batch_size = 2
n_train_samples = 50
n_test_samples = 10
n_features = 20
num_classes = config["num_classes"]

x_train_dummy = torch.randn(batch_size, n_train_samples, n_features)
y_train_dummy = torch.randint(0, num_classes, (batch_size, n_train_samples))
x_test_dummy = torch.randn(batch_size, n_test_samples, n_features)


3. Forward Pass (Inference)
You can run inference by calling the model with your data.

# --- Run a forward pass ---
with torch.no_grad():
    logits_output = model(x_train_dummy, y_train_dummy, x_test_dummy)

print("Output logits shape:", logits_output.shape)
# Expected: (2, 10, 10) -> (batch_size, n_test_samples, num_classes)

4. Training
To train the model, use the PyTorch Lightning Trainer. You will need to prepare a DataLoader that yields batches containing (x_train, y_train, x_test, y_test).

import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader

# Dummy test labels for the training step
y_test_dummy = torch.randint(0, num_classes, (batch_size, n_test_samples))

# This part requires a DataLoader setup
dataset = TensorDataset(x_train_dummy, y_train_dummy, x_test_dummy, y_test_dummy)
dataloader = DataLoader(dataset, batch_size=batch_size)

# --- Initialize and run trainer ---
trainer = pl.Trainer(max_epochs=5, accelerator='auto', logger=False)
trainer.fit(model, dataloader)

Citation
If you use this model in your research, please consider citing the original paper:

@article{qu2025tabicl,
  title={TabICL: A Tabular Foundation Model for In-Context Learning on Large Data},
  author={Qu, Jingang and Holzm{\"u}ller, David and Varoquaux, Ga{\"e}l and Le Morvan, Marine},
  journal={arXiv preprint arXiv:2502.05564},
  year={2025}
}
