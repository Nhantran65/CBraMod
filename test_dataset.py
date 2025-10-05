import torch
import numpy as np
from datasets.adhd_dataset import LoadDataset

class TestParams:
    def __init__(self):
        self.datasets_dir = "datasets/adhdata.csv"
        self.batch_size = 4

print("Testing dataset loading...")
try:
    params = TestParams()
    dataset = LoadDataset(params)
    data_loader = dataset.get_data_loader()
    
    # Test one batch from train loader
    train_loader = data_loader['train']
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Data shape: {data.shape}")
        print(f"  Target shape: {target.shape}")
        print(f"  Data type: {data.dtype}")
        print(f"  Target values: {target}")
        print(f"  Unique targets: {torch.unique(target)}")
        break
        
    print("✅ Dataset loading successful!")
    
except Exception as e:
    print(f"❌ Dataset loading failed: {e}")
    import traceback
    traceback.print_exc()