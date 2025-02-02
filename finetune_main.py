import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import math
from timeit import default_timer as timer
import random
from finetune_trainer import Trainer
from models import model_for_faced, model_for_seedv, model_for_physio, model_for_shu
from datasets import faced_dataset, seedv_dataset, physio_dataset, shu_dataset


def main():
    parser = argparse.ArgumentParser(description='Big model down stream')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=2, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    parser.add_argument('--downstream_dataset', type=str, default='PhysioNet-MI',
                        help='[FACED, SEED-V, PhysioNet-MI, SHU-MI]')
    parser.add_argument('--datasets_dir', type=str,
                        default='/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/processed_4_40hz',
                        help='datasets_dir')
    parser.add_argument('--num_of_classes', type=int, default=4, help='number of classes')
    parser.add_argument('--model_dir', type=str, default='/data/wjq/models_weights/Big/BigSEEDV', help='model_dir')

    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--frozen', type=bool,
                        default=False, help='frozen')
    parser.add_argument('--use_pretrained_weights', type=bool,
                        default=True, help='use_pretrained_weights')
    parser.add_argument('--foundation_dir', type=str,
                        # default='/data/wjq/models_weights/Big/0.4/epoch40_loss0.001386052928864956.pth',
                        default='pretrained_weights/pretrained_weights.pth',
                        help='foundation_dir')

    params = parser.parse_args()
    print(params)

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    print('The downstream dataset is {}'.format(params.downstream_dataset))
    if params.downstream_dataset == 'FACED':
        load_dataset = faced_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_faced.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'SEED-V':
        load_dataset = seedv_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_seedv.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'PhysioNet-MI':
        load_dataset = physio_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_physio.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'SHU-MI':
        load_dataset = shu_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_shu.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    print('Done!!!!!')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
