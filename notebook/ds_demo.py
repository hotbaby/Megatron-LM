# encoding: utf8

import time
import random
import argparse
import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import deepspeed
from deepspeed.profiling.flops_profiler import FlopsProfiler
from deepspeed import DeepSpeedEngine


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def get_device():
    if args.deepspeed:
        rank = torch.distributed.get_rank()
        return torch.device(rank)
    else:
        return torch.cuda.current_device()


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
    model = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=False), nn.GELU(), nn.Linear(hidden_dim, output_dim, bias=False))
    return model


args = get_args()

if args.deepspeed:
    deepspeed.init_distributed()

ds = DummyDataset()
model = get_model()

model.cuda(get_device())

if args.deepspeed:
    model_engine, optimzier, dataloader, _ = deepspeed.initialize(args, model=model, training_data=ds)
    model_engine: DeepSpeedEngine
else:
    dataloader = DataLoader(dataset=ds, batch_size=512)
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

criterion = nn.CrossEntropyLoss()


def train():
    total_iterations = len(dataloader)
    for i, (inputs, labels) in enumerate(dataloader):
        device = get_device()
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        if args.deepspeed:
            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)
            model_engine.backward(loss)
            model_engine.step()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"iteration {i+1}/{total_iterations}, loss: {loss:.2f}")


def benchmark_flops_profiler():
    device = get_device()
    inputs, labels = next(iter(dataloader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    prof = FlopsProfiler(model=model)
    prof.start_profile()

    model(inputs)

    total_flops = prof.get_total_flops(True)
    print(model)
    print(f"inputs: {inputs.size()}")
    print(f"total flops: {total_flops}")

    prof.end_profile()


if __name__ == "__main__":
    train()
    # benchmark_flops_profiler()
