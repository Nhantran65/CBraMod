# CBraMod ADHD Fine-tuning with K-Fold Cross Validation

Hướng dẫn chi tiết để chạy fine-tuning CBraMod trên dataset ADHD với 5-fold cross validation.

## 📋 Tổng quan

Notebook `kfold_adhd_finetune.ipynb` implement 5-fold stratified cross-validation để fine-tune mô hình CBraMod cho bài toán phân loại ADHD vs Control, sử dụng frozen backbone để tăng tốc training.

## 🎯 Tính năng chính

- ✅ **Stratified K-Fold**: Đảm bảo cân bằng classes (ADHD/Control) trong mỗi fold
- ✅ **Frozen Backbone**: Chỉ fine-tune classifier layer cuối (~3x faster)
- ✅ **Real Labels**: Sử dụng labels thật từ dataset (55.7% ADHD, 44.3% Control)
- ✅ **Flexible Dataset Size**: Chọn giữa full dataset (2.1M) hoặc sample dataset (10K)
- ✅ **Best Model Saving**: Tự động save best model cho mỗi fold
- ✅ **Comprehensive Evaluation**: Accuracy, F1-score, Cohen's Kappa
- ✅ **Detailed Visualization**: Training curves, confusion matrices, performance plots

## 🚀 Cách chạy

### 1. Environment Setup

```bash
# Cài đặt dependencies
conda install -c conda-forge einops
pip install torch torchvision torchaudio
pip install scikit-learn pandas matplotlib seaborn tqdm
```

### 2. Dataset Configuration

Mở file `kfold_adhd_finetune.ipynb`, tìm cell thứ 3 và chỉnh config:

```python
# 🎯 DATASET CONFIGURATION - CHOOSE YOUR OPTION
USE_FULL_DATASET = True   # True: 2.1M samples, False: sample dataset
SAMPLE_SIZE = 10000       # Số samples nếu dùng sample dataset
```

**Options:**

| Config | Dataset Size | Training Time | Performance | Recommended For |
|--------|-------------|---------------|-------------|-----------------|
| `USE_FULL_DATASET = True` | 2,166,383 samples | ~15-20 min | Highest accuracy | Final results |
| `USE_FULL_DATASET = False` | 10,000 samples | ~5-10 min | Good for testing | Quick experiments |

### 3. Chạy Notebook

**Chạy tuần tự các cells:**

1. **Cell 1**: Import libraries và setup seeds
2. **Cell 2**: Load dataset (chọn full hoặc sample)
3. **Cell 3**: Install missing packages (einops)
4. **Cell 4**: Load CBraMod model và config
5. **Cell 5**: Define training functions
6. **Cell 6**: **MAIN TRAINING** - 5-Fold Cross Validation
7. **Cell 7**: Results analysis và visualization

### 4. Monitor Training Progress

Trong quá trình training, bạn sẽ thấy:

```
🔥🔥🔥🔥 FOLD 1/5 🔥🔥🔥🔥
📊 Data Split:
  Train: 1,733,106 samples (ADHD: 965,655, Control: 767,451)
  Val:   433,277 samples (ADHD: 241,414, Control: 191,863)

🚀 Training Fold 1
============================================================
Epoch Train Loss  Train Acc   Val Loss    Val Acc     Best  
============================================================
1     0.6234      0.6543       0.5987       0.6123       ⭐ NEW
2     0.5876      0.6789       0.5654       0.6456       ⭐ NEW
...
🏆 Best Validation Accuracy: 0.6456
💾 Best model saved: ./saved_models/best_model_fold_1.pth
```

## 📊 Kết quả và Outputs

### Model Files
```
saved_models/
├── best_model_fold_1.pth          # Best model cho fold 1
├── best_model_fold_2.pth          # Best model cho fold 2
├── best_model_fold_3.pth          # Best model cho fold 3
├── best_model_fold_4.pth          # Best model cho fold 4
├── best_model_fold_5.pth          # Best model cho fold 5
├── cross_validation_results.png   # Visualization plots
└── cross_validation_summary.json  # Detailed results
```

### Performance Metrics
- **Mean Accuracy**: X.XXXX ± X.XXXX
- **Mean F1-Score**: X.XXXX ± X.XXXX  
- **Mean Kappa**: X.XXXX ± X.XXXX
- **Training Time**: X minutes per fold

### Visualizations
- Performance metrics across folds
- Box plots of metrics distribution  
- Training history curves
- Confusion matrices
- Training time comparison

## 🎯 Model Architecture

```
CBraMod Model:
├── Backbone (FROZEN)
│   ├── Patch Embedding: 19 channels → 200 dims
│   ├── Transformer Encoder: 12 layers, 8 heads
│   └── Pretrained weights: 4,883,800 parameters
└── Classifier (TRAINABLE)
    ├── AdaptiveAvgPool2d
    ├── Flatten  
    └── Linear: 200 → 2 classes
    └── Total trainable: 402 parameters
```

## ⚙️ Configuration

### Model Parameters
```python
class Args:
    num_subjects = 121
    window_size = 200
    encoding_dims = [256, 256]
    ff_hidden_size = 256
    attn_heads = 8
    depth = 12
    dropout_rate = 0.4
    freeze_backbone = True      # 🧊 Frozen for speed
    num_of_classes = 2          # ADHD vs Control
    use_pretrained_weights = True
```

### Training Parameters
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 64
- **Epochs per Fold**: 5
- **K-Fold Splits**: 5 (stratified)

## Dataset Format

File `datasets/adhdata.csv` cần có format:
```
Fp1,Fp2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T7,T8,P7,P8,Fz,Cz,Pz,Class,ID
261.0,402.0,16.0,...,ADHD,v10p
121.0,191.0,-94.0,...,Control,v11p
...
```

- 19 cột EEG channels: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz
- Class: "ADHD" hoặc "Control" (sẽ được encode thành 0, 1)
- ID: identifier cho mỗi sample

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```python
   # Giảm batch size trong notebook
   batch_size = 32  # thay vì 64
   ```

2. **DataLoader Error on Windows**:
   ```python
   # Đã fix: num_workers=0 cho Windows compatibility
   train_loader = DataLoader(..., num_workers=0)
   ```

3. **Module Not Found**:
   ```bash
   # Cài einops
   conda install -c conda-forge einops
   # hoặc
   pip install einops
   ```

### Memory Requirements
- **Full Dataset (2.1M)**: ~0.15 GB RAM
- **Sample Dataset (10K)**: ~0.001 GB RAM
- **GPU Memory**: ~2-4 GB (depending on batch size)

## 📈 Expected Performance

### With Full Dataset (2.1M samples):
- **Accuracy**: 65-75% (vs 50% random baseline)
- **Training Time**: 15-20 minutes total
- **Memory Usage**: 0.15 GB RAM

### With Sample Dataset (10K samples):
- **Accuracy**: 55-65% 
- **Training Time**: 5-10 minutes total
- **Memory Usage**: ~0.001 GB RAM

## 🎯 Tips for Best Results

1. **Use Full Dataset** cho maximum accuracy
2. **Monitor ⭐ NEW markers** để track best epochs
3. **Check class balance** trong mỗi fold
4. **Save results** trước khi close notebook
5. **Adjust sample size** nếu cần balance speed vs accuracy

## 📝 Files Structure

```
E:\Workspace\CBraMod\
├── kfold_adhd_finetune.ipynb     # Main notebook
├── ADHD_README.md                # This file
├── datasets/
│   └── adhdata.csv              # ADHD dataset (2.1M samples)
├── models/
│   └── model_for_adhd.py        # CBraMod model for ADHD
├── pretrained_weights/
│   └── pretrained_weights.pth   # Pretrained CBraMod weights
└── saved_models/                # Output models và results
    ├── best_model_fold_*.pth
    ├── cross_validation_results.png
    └── cross_validation_summary.json
```

## 🚀 Quick Start

```bash
# 1. Clone repo và navigate
cd E:\Workspace\CBraMod

# 2. Install dependencies  
conda install -c conda-forge einops

# 3. Open notebook
jupyter notebook kfold_adhd_finetune.ipynb

# 4. Chọn dataset size (cell 3)
USE_FULL_DATASET = True  # cho best results

# 5. Run all cells tuần tự
# 6. Wait ~15-20 minutes
# 7. Check results trong saved_models/
```

## 📞 Support

Nếu gặp vấn đề:
1. Check requirements đã install đủ chưa
2. Verify dataset path: `./datasets/adhdata.csv`
3. Check GPU memory với `nvidia-smi`
4. Restart kernel nếu cần

---

**Happy Fine-tuning with K-Fold Cross Validation! 🎉**