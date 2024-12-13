<div align="center">

# CBraMod


_A Criss-Cross Brain Foundation Model for EEG Decoding_

[![Paper](https://img.shields.io/badge/paper-2412.07236-red)](https://arxiv.org/abs/2412.07236)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/weighting666/CBraMod)
![GitHub Repo stars](https://img.shields.io/github/stars/wjq-learning/CBraMod)

</div>


<div align="center">
<img src="figure/CBraMod_logo.png" style="width: 15%;" />
</div>


## 🔍 About
We propose **CBraMod**, a novel EEG foundation model, for EEG decoding on various clinical and BCI application.
The preprint version of our paper is available at https://arxiv.org/abs/2412.07236.
<div align="center">
<img src="figure/model.png" style="width:80%;" />
</div>

## 🔧 The repository is updating ...


## 🔥 How to Pretrain
You can pretrain CBraMod on our pretraining dataset or your custom pretraining dataset using the following code:
```bash
python pretrain_main.py
```


## 🚀 Quick Start
You can fine-tune the pretrained CBraMod on your custom downstream dataset using the following example code:
```python
import torch
import torch.nn as nn
from models.cbramod import CBraMod
from einops.layers.torch import Rearrange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CBraMod().to(device)
model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth', map_location=device))
classifier = nn.Sequential(
  Rearrange('b c s p -> b (c s p)'),
  nn.Linear(22*4*200, 4*200),
  nn.GELU(),
  nn.Dropout(0.1),
  nn.Linear(4 * 200, 200),
  nn.GELU(),
  nn.Dropout(0.1),
  nn.Linear(200, 4),
).to(device)

# mock_eeg.shape = (batch_size, num_of_channels, time_segments, points_per_patch)
mock_eeg = torch.randn((8, 22, 4, 200)).to(device)

# logits.shape = (batch_size, num_of_classes)
logits = classifier(model(mock_eeg))
```



## 🔗 Citation
If you're using this repository in your research or applications, please cite using the following BibTeX:
```bibtex
@article{wang2024cbramod,
  title={CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding},
  author={Wang, Jiquan and Zhao, Sha and Luo, Zhiling and Zhou, Yangxuan and Jiang, Haiteng and Li, Shijian and Li, Tao and Pan, Gang},
  journal={arXiv preprint arXiv:2412.07236},
  year={2024}
}
```