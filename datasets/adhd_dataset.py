import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from utils.util import to_tensor
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class ADHDDataset(Dataset):
    def __init__(self, data, labels):
        super(ADHDDataset, self).__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label).long()


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir
        self.prepare_data()

    def prepare_data(self):
        """
        Prepare ADHD dataset from CSV file - REAL LABELS, NO SYNTHETIC
        """
        csv_path = self.datasets_dir
        print(f"Loading data from: {csv_path}")
        
        # Get total number of rows first
        print("Counting total rows...")
        total_rows = sum(1 for line in open(csv_path)) - 1  # -1 for header
        print(f"Total data rows: {total_rows:,}")
        
        # Random sampling to get balanced representation
        target_samples = 200000  # Reduced for memory efficiency
        if total_rows > target_samples:
            # Calculate skip pattern for random sampling
            np.random.seed(42)  # Reproducible sampling
            skip_rows = sorted(np.random.choice(range(1, total_rows + 1), 
                                              size=total_rows - target_samples, 
                                              replace=False))
            print(f"Random sampling {target_samples:,} from {total_rows:,} rows...")
            df = pd.read_csv(csv_path, skiprows=skip_rows)
        else:
            print("Loading all data...")
            df = pd.read_csv(csv_path)
        
        print(f"Loaded dataset shape: {df.shape}")
        
        # Use actual column order from dataset (not the documentation order)
        eeg_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                       'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']
        
        # Extract EEG features
        X = df[eeg_channels].values.astype(np.float32)
        
        # Use REAL labels - not synthetic!
        y = df['Class'].values
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        print(f"Real classes: {label_encoder.classes_}")
        print(f"Class distribution: {np.bincount(y_encoded)}")
        class_percentages = np.bincount(y_encoded) / len(y_encoded) * 100
        for i, cls in enumerate(label_encoder.classes_):
            print(f"  {cls}: {np.bincount(y_encoded)[i]:,} samples ({class_percentages[i]:.1f}%)")
        print(f"Number of samples: {len(X):,}")
        print(f"Unique patient IDs: {df['ID'].nunique()}")
        
        # Normalize the EEG data (z-score normalization)
        X_mean = np.mean(X, axis=0, keepdims=True)
        X_std = np.std(X, axis=0, keepdims=True) + 1e-8  # Add small epsilon to avoid division by zero
        X = (X - X_mean) / X_std
        
        # Reshape data for the model (samples, channels, time_points)
        # Since we have single time points, we'll expand to create a time dimension
        X = np.expand_dims(X, axis=-1)  # Shape: (samples, channels, 1)
        
        # Split data into train/val/test with balanced classes
        try:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 * 0.8 = 0.2 for val
            )
        except ValueError as e:
            print(f"Warning: Could not stratify split (probably imbalanced classes): {e}")
            print("Using random split instead...")
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, random_state=42
            )
        
        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        # Store the splits
        self.train_data = X_train
        self.train_labels = y_train
        self.val_data = X_val
        self.val_labels = y_val
        self.test_data = X_test
        self.test_labels = y_test
        
        # Store label encoder for later use
        self.label_encoder = label_encoder
        self.classes = label_encoder.classes_
        self.num_classes = len(label_encoder.classes_)

    def get_data_loader(self):
        train_set = ADHDDataset(self.train_data, self.train_labels)
        val_set = ADHDDataset(self.val_data, self.val_labels)
        test_set = ADHDDataset(self.test_data, self.test_labels)
        
        print(f"Dataset sizes - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
        print(f"Total samples: {len(train_set) + len(val_set) + len(test_set)}")
        
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                shuffle=True,
                num_workers=0,  # Set to 0 for Windows compatibility
                collate_fn=train_set.collate,
                drop_last=True
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=val_set.collate,
                drop_last=False
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=test_set.collate,
                drop_last=False
            )
        }
        return data_loader