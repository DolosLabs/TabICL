import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import requests
import math

# =============================================================================
# ## 1. Model Definition (TabICL and its components)
# This section contains the full model architecture, including all custom
# transformer blocks and the main LightningModule.
# =============================================================================

# ================================================================
# Rotary Positional Embedding (RoPE) utilities
# ================================================================
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = -1
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, seq_len, device):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        return self.cos_cached, self.sin_cached

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

# ================================================================
# Custom Transformer Components (RoPE-enabled)
# ================================================================
class RotaryMultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        b, seq_len, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = q.view(b, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(b, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(b, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(seq_len, x.device)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(b, seq_len, self.d_model)
        return self.out_proj(attn_output)

class TransformerEncoderLayerCustom(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        self.self_attn = RotaryMultiHeadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class TransformerEncoderCustom(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])
    def forward(self, src, mask=None, key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask=mask, src_key_padding_mask=key_padding_mask)
        return src

# ================================================================
# Set Transformer components (TF_col)
# ================================================================
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q, K, V = self.fc_q(Q), self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = self.ln0(O) if hasattr(self, 'ln0') else O
        O = O + F.relu(self.fc_o(O))
        O = self.ln1(O) if hasattr(self, 'ln1') else O
        return O

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class SetTransformer(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, num_blocks):
        super().__init__()
        self.enc = nn.Sequential(
            ISAB(dim_in, dim_out, num_heads, num_inds),
            *[ISAB(dim_out, dim_out, num_heads, num_inds) for _ in range(num_blocks - 1)]
        )
        self.dec1 = nn.Linear(dim_out, dim_out)
        self.dec2 = nn.Linear(dim_out, dim_out)

    def forward(self, X):
        X_emb = self.enc(X)
        return X * self.dec1(X_emb) + self.dec2(X_emb)

# ================================================================
# Main Model Components
# ================================================================
class RowInteractionTransformer(nn.Module):
    def __init__(self, dim, n_layers, n_heads, dim_feedforward, dropout=0.0):
        super().__init__()
        self.cls_count = 4
        self.cls_tokens = nn.Parameter(torch.randn(1, self.cls_count, dim))
        layer = TransformerEncoderLayerCustom(
            d_model=dim, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.encoder = TransformerEncoderCustom(layer, num_layers=n_layers)

    def forward(self, x, key_padding_mask=None):
        b = x.shape[0]
        cls_tokens = self.cls_tokens.expand(b, -1, -1)
        seq = torch.cat((cls_tokens, x), dim=1)
        if key_padding_mask is not None:
            cls_mask = torch.zeros(b, self.cls_count, dtype=torch.bool, device=seq.device)
            key_padding_mask = torch.cat((cls_mask, key_padding_mask), dim=1)
        enc = self.encoder(seq, mask=None, key_padding_mask=key_padding_mask)
        return enc[:, :self.cls_count, :].reshape(b, -1)

class ICLTransformer(nn.Module):
    # --- UPDATED ---
    # Simplified this class to work for a single dataset, as required by the training script.
    # It now takes `num_classes` directly and uses a single output head.
    def __init__(self, d_model, nhead, num_layers, hidden_dim, dropout, num_classes):
        super().__init__()
        # Note: PyTorch's default Transformer expects dimensions (Seq, Batch, Feat)
        # We will handle this by transposing the input tensor.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, num_classes)

    # --- UPDATED ---
    # Simplified the forward pass signature. It now only takes support and query sets.
    # The `support_y` and `dataset_name` arguments were removed as they were not used
    # or were incompatible with a single-dataset setup.
    def forward(self, support_x, query_x):
        # Concatenate support & query sets along the sequence dimension
        x = torch.cat([support_x, query_x], dim=1)  # Shape: [B, N_support + N_query, D]

        # Transformer Encoder expects [SeqLen, Batch, Dim] by default
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # Back to [Batch, SeqLen, Dim]

        # Extract only the representations for the query set
        support_len = support_x.size(1)
        query_repr = x[:, support_len:, :]  # Shape: [B, n_query, D]

        logits = self.output_head(query_repr)
        return logits

# ================================================================
# TabICL LightningModule
# ================================================================
class TabICL(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # self.hparams is deprecated; use self.hparams from save_hyperparameters
        self.save_hyperparameters(config)
        hparams = self.hparams

        self.tf_col = SetTransformer(
            dim_in=1, dim_out=hparams.d_model, num_heads=hparams.tf_col_heads,
            num_inds=hparams.tf_col_inds, num_blocks=hparams.tf_col_blocks
        )
        self.tf_row = RowInteractionTransformer(
            dim=hparams.d_model, n_layers=hparams.tf_row_layers, n_heads=hparams.tf_row_heads,
            dim_feedforward=hparams.tf_row_ff, dropout=hparams.dropout
        )
        # --- UPDATED ---
        # The initialization now correctly matches the simplified ICLTransformer
        self.tf_icl = ICLTransformer(
            d_model=hparams.d_model * self.tf_row.cls_count, # d_model * 4
            nhead=hparams.tf_icl_heads,
            num_layers=hparams.tf_icl_layers,
            hidden_dim=hparams.tf_icl_ff,
            num_classes=hparams.num_classes,
            dropout=hparams.dropout
        )
        
    def forward(self, x_train, x_test, y_train=None, train_padding_mask=None, test_padding_mask=None):
        # y_train is passed by the training step but is not used in this forward pass, so we accept it.
        
        # --- Get input shapes ---
        batch_size, n_train, n_features = x_train.shape
        _, n_test, _ = x_test.shape
        x_full = torch.cat((x_train, x_test), dim=1)
        n_full = n_train + n_test

        # --- 1. Vectorized Column-wise Embedding ---
        x_reshaped = x_full.permute(0, 2, 1).reshape(batch_size * n_features, n_full, 1)
        embedded_cols = self.tf_col(x_reshaped)
        d_model = embedded_cols.shape[-1]
        
        E = embedded_cols.reshape(batch_size, n_features, n_full, d_model).permute(0, 2, 1, 3)
        
        # --- 2. Vectorized Row-wise Interaction ---
        E_reshaped = E.reshape(batch_size * n_full, n_features, d_model)

        if train_padding_mask is not None and test_padding_mask is not None:
            full_padding_mask = torch.cat((train_padding_mask, test_padding_mask), dim=1)
            padding_mask_reshaped = full_padding_mask.reshape(batch_size * n_full, n_features)
        else:
            padding_mask_reshaped = None

        H_reshaped = self.tf_row(E_reshaped, key_padding_mask=padding_mask_reshaped)
        H = H_reshaped.reshape(batch_size, n_full, -1)
        
        # --- 3. In-Context Learning ---
        H_train = H[:, :n_train, :]
        H_test = H[:, n_train:, :]

        # --- UPDATED ---
        # Corrected the call to match the updated ICLTransformer.forward()
        return self.tf_icl(H_train, H_test)

    def _calculate_metrics(self, batch, stage):
        # Unpack batch
        x_train, y_train, x_test, y_test = batch
        
        # The forward pass expects x_train and x_test. y_train is ignored inside but passed for signature compatibility.
        logits = self(x_train, x_test, y_train)
        
        loss = F.cross_entropy(logits.view(-1, self.hparams.num_classes), y_test.view(-1))
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y_test).float().mean()
        
        self.log(f'{stage}_loss', loss, prog_bar=True, on_step=(stage=='train'), on_epoch=True)
        self.log(f'{stage}_acc', acc, prog_bar=True, on_step=(stage=='train'), on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_metrics(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self._calculate_metrics(batch, 'val')

    def test_step(self, batch, batch_idx):
        self._calculate_metrics(batch, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.get("weight_decay", 0.01)
        )
        return optimizer

# =============================================================================
# ## 2. Data Loading and Preparation
# This section defines how to download, process, and serve the data.
# =============================================================================

# This dataset generates "tasks" or "prompts" for in-context learning.
# Each item is a tuple of (support_set_X, support_set_y, query_set_X, query_set_y)
class ICLDataset(Dataset):
    def __init__(self, X, y, n_train_samples, n_test_samples, num_tasks):
        self.X = X
        self.y = y
        self.n_train_samples = n_train_samples # In-context examples
        self.n_test_samples = n_test_samples   # Examples to predict
        self.num_tasks = num_tasks             # Number of prompts per epoch

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, idx):
        total_samples = self.n_train_samples + self.n_test_samples
        
        # Randomly sample a small batch of data for this specific prompt
        indices = np.random.choice(len(self.X), total_samples, replace=False)
        
        # Split into a support set (context) and a query set
        train_indices = indices[:self.n_train_samples]
        test_indices = indices[self.n_train_samples:]
        
        x_train = self.X[train_indices]
        y_train = self.y[train_indices]
        x_test = self.X[test_indices]
        y_test = self.y[test_indices]
        
        return x_train, y_train, x_test, y_test

# The LightningDataModule handles the entire data lifecycle.
class AdultDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 32, 
                 n_train_samples: int = 40, n_test_samples: int = 10,
                 tasks_per_epoch: int = 1000,
                 num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self.num_workers = num_workers
        self.tasks_per_epoch = tasks_per_epoch 
        self.preprocessor = None
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None

    def prepare_data(self):
        # Download data if it's not already there
        urls = {
            "train": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            "test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
        }
        os.makedirs(self.data_dir, exist_ok=True)
        for split, url in urls.items():
            filepath = os.path.join(self.data_dir, f"adult.{split}")
            if not os.path.exists(filepath):
                print(f"Downloading {split} data...")
                r = requests.get(url, allow_redirects=True)
                open(filepath, 'wb').write(r.content)

    def setup(self, stage: str = None):
        columns = [
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"
        ]
        
        # Load data using pandas
        df_train = pd.read_csv(os.path.join(self.data_dir, "adult.train"), header=None, names=columns, 
                               na_values=" ?", sep=r',\s*', engine='python')
        df_test = pd.read_csv(os.path.join(self.data_dir, "adult.test"), header=None, names=columns, 
                              na_values=" ?", sep=r',\s*', engine='python', skiprows=1)
        
        df = pd.concat([df_train, df_test], ignore_index=True)
        
        # Preprocess the data: drop missing values, encode target, etc.
        df.dropna(inplace=True)
        df['income'] = df['income'].apply(lambda x: 1 if x.startswith('>50K') else 0)

        X = df.drop('income', axis=1)
        y = df['income']

        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

        # Create a scikit-learn pipeline to scale numerical and one-hot encode categorical features
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ])

        X_processed = self.preprocessor.fit_transform(X)

        # Split into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X_processed, y.values, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        # Convert final datasets to PyTorch tensors
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.X_val = torch.tensor(X_val, dtype=torch.float32)
        self.y_val = torch.tensor(y_val, dtype=torch.long)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.long)
        
        print(f"Data is ready. Shapes: Train={self.X_train.shape}, Val={self.X_val.shape}, Test={self.X_test.shape}")

    def _get_dataloader(self, X, y, num_tasks):
        dataset = ICLDataset(X, y, self.n_train_samples, self.n_test_samples, num_tasks)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def train_dataloader(self):
        return self._get_dataloader(self.X_train, self.y_train, num_tasks=self.tasks_per_epoch)

    def val_dataloader(self):
        return self._get_dataloader(self.X_val, self.y_val, num_tasks=self.tasks_per_epoch // 5)

    def test_dataloader(self):
        return self._get_dataloader(self.X_test, self.y_test, num_tasks=self.tasks_per_epoch // 5)


# =============================================================================
# ## 3. Main Training Execution 🚀
# This is the main entry point of the script.
# =============================================================================
if __name__ == '__main__':
    pl.seed_everything(42)

    # --- Configuration ---
    # Model hyperparameters and training settings
    config = {
        "d_model": 64,          # Embedding dimension
        "num_classes": 2,       # Binary classification: <=50K or >50K
        "learning_rate": 5e-5,
        "dropout": 0.1,
        # TF_col (SetTransformer for columns)
        "tf_col_heads": 4,
        "tf_col_inds": 16,      # Number of inducing points
        "tf_col_blocks": 2,
        # TF_row (Transformer for rows)
        "tf_row_layers": 2,
        "tf_row_heads": 4,
        "tf_row_ff": 128,
        # TF_icl (Transformer for In-Context Learning)
        "tf_icl_layers": 2,
        "tf_icl_heads": 4,
        "tf_icl_ff": 256,
    }
    
    # --- Set up Data Module ---
    # Each item in a batch consists of 40 "in-context" examples and 10 "query" examples
    data_module = AdultDataModule(
        batch_size=64, 
        n_train_samples=40, 
        n_test_samples=10,
        tasks_per_epoch=640,
        num_workers=4
    )

    # --- Initialize Model ---
    model = TabICL(config)

    # --- Initialize Trainer ---
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto", # Auto-selects hardware like GPU or CPU
        devices=1,
        log_every_n_steps=10,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=3),
            pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
        ]
    )

    # --- Run Training and Testing ---
    print("\n--- Starting Training ---")
    trainer.fit(model, datamodule=data_module)
    
    print("\n--- Starting Testing ---")
    # The trainer automatically loads the best model checkpoint for testing
    trainer.test(model, datamodule=data_module)