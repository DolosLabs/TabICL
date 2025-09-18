import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math

# Helper module for Rotary Positional Embedding (RoPE)
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]

        return self.cos_cached, self.sin_cached

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

# Multi-head Attention Block (MAB) - as per SetTransformer
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
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
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

# Induced Self-Attention Block (ISAB) - as per SetTransformer
class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

# Set Transformer for Column-wise embedding (TF_col)
class SetTransformer(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, num_blocks):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_in, dim_out, num_heads, num_inds),
            *[ISAB(dim_out, dim_out, num_heads, num_inds) for _ in range(num_blocks - 1)]
        )
        self.dec1 = nn.Linear(dim_out, dim_out)
        self.dec2 = nn.Linear(dim_out, dim_out)


    def forward(self, X):
        # Expects X to be (batch_size, num_samples, 1) for a single column
        X_emb = self.enc(X)
        W = self.dec1(X_emb)
        b = self.dec2(X_emb)
        return X * W + b

# Row-wise Interaction Transformer (TF_row)
class RowInteractionTransformer(nn.Module):
    def __init__(self, dim, n_layers, n_heads, dim_feedforward):
        super().__init__()
        self.rotary_emb = RotaryPositionalEmbedding(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_tokens = nn.Parameter(torch.randn(1, 4, dim))

    def forward(self, x):
        # x is expected to be (batch_size, num_features, dim)
        b, n, d = x.shape
        cls_tokens = self.cls_tokens.repeat(b, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)

        # RoPE is typically applied inside the attention mechanism.
        # For simplicity in this implementation using nn.TransformerEncoderLayer,
        # we demonstrate its application conceptually.
        # A custom attention layer would be needed for precise RoPE integration.
        # This is a placeholder for where RoPE would be applied.
        # cos, sin = self.rotary_emb(x)
        # q, k = apply_rotary_pos_emb(q, k, cos, sin) # in attention
        
        output = self.transformer_encoder(x)
        
        # Concatenate the outputs of the CLS tokens
        return output[:, :4, :].reshape(b, -1)


# In-Context Learning Transformer (TF_icl)
class ICLTransformer(nn.Module):
    def __init__(self, dim, n_layers, n_heads, dim_feedforward, num_classes):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_classes)
        )
        self.label_embedding = nn.Embedding(num_classes, dim)

    def forward(self, train_embeds, test_embeds, train_labels):
        # train_embeds: (batch, n_train, dim)
        # test_embeds: (batch, n_test, dim)
        # train_labels: (batch, n_train)
        
        label_embeds = self.label_embedding(train_labels)
        train_input = train_embeds + label_embeds
        
        full_seq = torch.cat((train_input, test_embeds), dim=1)
        
        # Create attention mask to prevent test samples from attending to each other
        n_train = train_embeds.size(1)
        n_test = test_embeds.size(1)
        seq_len = n_train + n_test
        mask = torch.zeros(seq_len, seq_len, device=train_embeds.device).bool()
        mask[n_train:, n_train:] = True # Test samples can't see each other
        
        output = self.transformer_encoder(full_seq, mask=mask)
        
        test_output = output[:, n_train:, :]
        return self.mlp_head(test_output)


class TabICL(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        # 1. Column-wise Embedding (TF_col)
        self.tf_col = SetTransformer(
            dim_in=1, # Each cell is a scalar
            dim_out=self.hparams.d_model,
            num_heads=self.hparams.tf_col_heads,
            num_inds=self.hparams.tf_col_inds,
            num_blocks=self.hparams.tf_col_blocks
        )

        # 2. Row-wise Interaction (TF_row)
        self.tf_row = RowInteractionTransformer(
            dim=self.hparams.d_model,
            n_layers=self.hparams.tf_row_layers,
            n_heads=self.hparams.tf_row_heads,
            dim_feedforward=self.hparams.tf_row_ff
        )

        # 3. In-Context Learning (TF_icl)
        self.tf_icl = ICLTransformer(
            dim=self.hparams.d_model * 4, # 4 CLS tokens concatenated
            n_layers=self.hparams.tf_icl_layers,
            n_heads=self.hparams.tf_icl_heads,
            dim_feedforward=self.hparams.tf_icl_ff,
            num_classes=self.hparams.num_classes
        )
        
    def forward(self, x_train, y_train, x_test):
        # x_train: (batch, n_train, n_features)
        # y_train: (batch, n_train)
        # x_test: (batch, n_test, n_features)
        
        batch_size, n_train, n_features = x_train.shape
        _, n_test, _ = x_test.shape

        # Combine train and test for efficient feature embedding
        x_full = torch.cat((x_train, x_test), dim=1)
        n_full = n_train + n_test

        # Process each feature column
        col_embeds = []
        for i in range(n_features):
            col = x_full[:, :, i].unsqueeze(-1) # (batch, n_full, 1)
            # In the paper, this is done per-dataset. Here we batch over datasets.
            # Assuming batch here means multiple datasets.
            embedded_col = self.tf_col(col) # (batch, n_full, d_model)
            col_embeds.append(embedded_col)
        
        # E in paper: (batch, n_full, n_features, d_model)
        E = torch.stack(col_embeds, dim=2)
        
        # Row-wise interaction
        row_embeds_list = []
        # Process each row
        for i in range(n_full):
            row_input = E[:, i, :, :] # (batch, n_features, d_model)
            row_embed = self.tf_row(row_input) # (batch, d_model * 4)
            row_embeds_list.append(row_embed)
            
        # H in paper: (batch, n_full, d_model * 4)
        H = torch.stack(row_embeds_list, dim=1)

        H_train = H[:, :n_train, :]
        H_test = H[:, n_train:, :]
        
        # In-Context Learning
        logits = self.tf_icl(H_train, H_test, y_train)
        return logits

    def training_step(self, batch, batch_idx):
        x_train, y_train, x_test, y_test = batch
        logits = self(x_train, y_train, x_test)
        loss = F.cross_entropy(logits.view(-1, self.hparams.num_classes), y_test.view(-1))
        self.log('train_loss', loss)
        
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y_test).float().mean()
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

if __name__ == '__main__':
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

    # --- Dummy Data for demonstration ---
    batch_size = 2
    n_train_samples = 50
    n_test_samples = 10
    n_features = 20
    
    x_train_dummy = torch.randn(batch_size, n_train_samples, n_features)
    y_train_dummy = torch.randint(0, config["num_classes"], (batch_size, n_train_samples))
    
    x_test_dummy = torch.randn(batch_size, n_test_samples, n_features)
    y_test_dummy = torch.randint(0, config["num_classes"], (batch_size, n_test_samples))

    # --- Run a forward pass ---
    print("\n--- Running a forward pass ---")
    with torch.no_grad():
        logits_output = model(x_train_dummy, y_train_dummy, x_test_dummy)
    
    print("Input x_test shape:", x_test_dummy.shape)
    print("Output logits shape:", logits_output.shape)
    print("Expected output shape:", (batch_size, n_test_samples, config["num_classes"]))
    assert logits_output.shape == (batch_size, n_test_samples, config["num_classes"])
    print("Forward pass successful!")

    # --- Example of setting up a PyTorch Lightning Trainer ---
    # This part requires a DataLoader setup, which is omitted for simplicity.
    # from torch.utils.data import TensorDataset, DataLoader
    #
    # dataset = TensorDataset(x_train_dummy, y_train_dummy, x_test_dummy, y_test_dummy)
    # dataloader = DataLoader(dataset, batch_size=batch_size)
    #
    # trainer = pl.Trainer(max_epochs=1, accelerator='auto')
    # trainer.fit(model, dataloader)
