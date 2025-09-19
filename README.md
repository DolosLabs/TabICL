# TabICL: A PyTorch Lightning Implementation

This repository contains a **PyTorch Lightning** implementation of the **TabICL** model, as described in the paper:

> Qu et al. (2025), *"TabICL: A Tabular Foundation Model for In-Context Learning on Large Data"*.

The implementation is self-contained within **`tabicl_lightning.py`** for ease of use.

---

## 📐 Model Architecture

The model is composed of three main transformer-based stages, following the architecture outlined in the paper:

### 1. Distribution-aware Column-wise Embedding (**TF_col**)
- Implemented as **SetTransformer**.
- Processes each feature column independently.
- Uses **Induced Self-Attention Blocks (ISAB)** to capture statistical properties of each column and create powerful feature embeddings.

### 2. Context-aware Row-wise Interaction (**TF_row**)
- Implemented as **RowInteractionTransformer**.
- Takes the feature embeddings for each row and prepends learnable `[CLS]` tokens.
- Uses a **transformer encoder** to model inter-feature dependencies.
- Supports **Rotary Positional Embeddings (RoPE)** for distinguishing identically distributed features (placeholder included).
- Final row embedding = concatenation of the four `[CLS]` tokens.

### 3. Dataset-wise In-Context Learning (**TF_icl**)
- Implemented as **ICLTransformer**.
- Performs the final prediction step.
- Takes embedded training rows (with label embeddings) + embedded test rows as a single sequence.
- Uses a **causal attention mask** to predict classes for all test samples in one forward pass, leveraging training samples as in-context examples.

---

## 📂 File Structure

- **`tabicl_lightning.py`** – contains everything:
  - Helper modules (`RotaryPositionalEmbedding`, `MAB`, `ISAB`)
  - Core components (`SetTransformer`, `RowInteractionTransformer`, `ICLTransformer`)
  - Main `TabICL` LightningModule (orchestrates training/logic)
  - Example `if __name__ == "__main__":` block with dummy data
- **`train.py`** – example training script using PyTorch Lightning.

---

## ⚙️ Requirements

- Python 3.x  
- PyTorch  
- PyTorch Lightning  

Install via pip:

```bash
pip install torch pytorch-lightning
````

---

## 🚀 Usage

### 1. Quickstart with `train.py`

Run the provided training script:

```bash
python train.py
```

This will:

* Initialize a `TabICL` model with default configs.
* Generate dummy data (`x_train`, `y_train`, `x_test`, `y_test`).
* Train the model for a few epochs using PyTorch Lightning.

---

### 2. Model Initialization (Manual)

```python
from tabicl_lightning import TabICL

# --- Configuration ---
config = {
    "d_model": 128,
    "num_classes": 10,
    "learning_rate": 1e-4,

    # TF_col
    "tf_col_heads": 4,
    "tf_col_inds": 32,   # Number of inducing points
    "tf_col_blocks": 2,

    # TF_row
    "tf_row_layers": 3,
    "tf_row_heads": 8,
    "tf_row_ff": 512,

    # TF_icl
    "tf_icl_layers": 12,
    "tf_icl_heads": 4,
    "tf_icl_ff": 1024,
}

model = TabICL(config)
print(model)
```

---

### 3. Data Preparation

The model expects data in the following format:

* **`x_train`**: `(batch_size, n_train_samples, n_features)` – training features
* **`y_train`**: `(batch_size, n_train_samples)` – training labels
* **`x_test`**: `(batch_size, n_test_samples, n_features)` – test features

```python
import torch

# --- Dummy Data for demonstration ---
batch_size = 2
n_train_samples = 50
n_test_samples = 10
n_features = 20
num_classes = config["num_classes"]

x_train_dummy = torch.randn(batch_size, n_train_samples, n_features)
y_train_dummy = torch.randint(0, num_classes, (batch_size, n_train_samples))
x_test_dummy  = torch.randn(batch_size, n_test_samples, n_features)
```

---

### 4. Forward Pass (Inference)

```python
# --- Run a forward pass ---
with torch.no_grad():
    logits_output = model(x_train_dummy, y_train_dummy, x_test_dummy)

print("Output logits shape:", logits_output.shape)
# Expected: (2, 10, 10) -> (batch_size, n_test_samples, num_classes)
```

---

### 5. Training (Custom)

To train the model, use the **PyTorch Lightning Trainer**.
You’ll need a DataLoader that yields `(x_train, y_train, x_test, y_test)`.

```python
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader

# Dummy test labels
y_test_dummy = torch.randint(0, num_classes, (batch_size, n_test_samples))

# Dataset + DataLoader
dataset = TensorDataset(x_train_dummy, y_train_dummy, x_test_dummy, y_test_dummy)
dataloader = DataLoader(dataset, batch_size=batch_size)

# --- Initialize and run trainer ---
trainer = pl.Trainer(max_epochs=5, accelerator='auto', logger=False)
trainer.fit(model, dataloader)
```

---

## 📖 Citation

If you use this model in your research, please cite the original paper:

```bibtex
@article{qu2025tabicl,
  title={TabICL: A Tabular Foundation Model for In-Context Learning on Large Data},
  author={Qu, Jingang and Holzm{\"u}ller, David and Varoquaux, Ga{\"e}l and Le Morvan, Marine},
  journal={arXiv preprint arXiv:2502.05564},
  year={2025}
}
```