import torch
import torch.nn as nn
from functools import partial
from torcheeg.models import EEGNet, FBCCNN, VanillaTransformer, TSCeption, STNet, CCNN, LSTM, ArjunViT, DGCNN, LGGNet
from torcheeg.models.pyg import *
from torcheeg.datasets.constants.emotion_recognition.deap import DEAP_GENERAL_REGION_LIST, DEAP_STANDARD_ADJACENCY_MATRIX
from .cbramod import CBraMod



class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )
        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))
        self.backbone.proj_out = nn.Sequential()
        self.classifier = nn.Sequential(
            nn.Linear(64*3*200, 3*200),
            nn.GELU(),
            nn.Linear(3*200, 200),
            nn.GELU(),
            nn.Linear(200, param.num_of_classes)
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        feats = feats.contiguous().view(bz, ch_num*seq_len*200)
        out = self.classifier(feats)
        return out

