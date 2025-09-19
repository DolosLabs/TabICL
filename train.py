import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import requests

# Import the model from your separate file
from TabICL import TabICL

# =============================================================================
# ## 1. Data Loading and Preparation
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
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        X_processed = self.preprocessor.fit_transform(X).toarray()

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
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def train_dataloader(self):
        # Use the new parameter instead of a hardcoded value
        return self._get_dataloader(self.X_train, self.y_train, num_tasks=self.tasks_per_epoch)

    # You can also adjust validation/test tasks if you want
    def val_dataloader(self):
        return self._get_dataloader(self.X_val, self.y_val, num_tasks=self.tasks_per_epoch // 5)

    def test_dataloader(self):
        return self._get_dataloader(self.X_test, self.y_test, num_tasks=self.tasks_per_epoch // 5)

# =============================================================================
# ## 2. Main Training Execution 🚀
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
        "dropout": .1,
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
    # The model is imported from TabICL.py and initialized with our config
    model = TabICL(config)

    # --- Initialize Trainer ---
    # The PyTorch Lightning Trainer automates the training process
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto", # Auto-selects hardware like GPU or CPU
        devices=1,
        log_every_n_steps=5,
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