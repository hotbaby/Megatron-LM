# encoding: utf8

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import deepspeed


def get_args():
    parser = argparse.ArgumentParser()
    deepspeed.add_config_arguments(parser)
    
    args = parser.parse_args()
    return args


class DummyDataset(Dataset):
    
    def __init__(self, input_dim=1024, vocab_size=1024, **kwargs):
        data_size = 10240
        self.inputs = torch.randn(data_size, input_dim)
        self.labels = torch.empty(data_size, dtype=torch.long).random_(vocab_size)
    
    def __getitem__(self, index):
        return self.inputs[index,:], self.labels[index]

    def __len__(self):
        return len(self.labels)


def get_model(input_dim=1024, output_dim=1024, hidden_dim=2048):
    model = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))
    model.cuda()
    return model


args = get_args()
deepspeed.init_distributed()
model = get_model()
model_engine, optimzier, _, _ = deepspeed.initialize(args, model=model)

criterion = nn.CrossEntropyLoss()


def train():
    pass
