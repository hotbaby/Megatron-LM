# encoding: utf8
import os
import re
import json
import random
import numpy as np
import pandas as pd
import dataclasses
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from typing import List, Dict


# 环境变量
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"


#模型和训练超参数
@dataclasses.dataclass
class LMConfig:
    num_layers: int
    hidden_size: int
    num_attention_head: int
    seq_length: int = 1024


@dataclasses.dataclass
class Config:
    gpus_per_node: int

    # train configuration
    steps: int
    batch_size: int
    # 混合精度训练
    fp16: bool
    
    # model configuration
    lm: LMConfig
    
    # parallel configuration
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # activation checkpoint
    activation_ckpt: bool = False
    
    # Gradient Accumulation
    grad_accum: bool = False

    # 节点数
    nnodes: int = 1


@dataclasses.dataclass
class Metrics:
    forward_backward: float
    forward: float
    backward: float
    optimizer: float
    communication: float
    memory: float
    samples_per_second: float


def print_metrics(metrics: Metrics, scheme: str):
    compute = metrics.forward_backward + metrics.optimizer
    communication = metrics.communication

    print(f"{scheme}: {json.dumps(dataclasses.asdict(metrics))}")


# metrics = Metrics(forward_backward=300, forward=100, backward=200, communication=10,
#                   optimizer=10, memory=200, samples_per_second=2)
# print_metrics(metrics, "base")


# 绘图函数
def plot(data: pd.DataFrame):
    ncols = len(data.columns)
    x = list(data.index)
    fig, axs = plt.subplots(ncols=ncols, figsize=(16, 2.4))
    fig.subplots_adjust(wspace=0.42)

    for i, metric in enumerate(data.columns):
        y = list(map(lambda x: float(x), data[metric].values))
        axs[i].barh(x, y, height=0.8)
        axs[i].set_xlabel(metric)

    plt.show()

# 测试plot函数
# data = {scheme: {"compute": random.randint(1, 10), 
# #                  "optimizer": random.randint(1, 10), 
# #                  "memory": random.randint(10,100), 
#                  "communication": random.randint(100, 1000)} for scheme in ["base", "fp16", "mp=2,\npp=2"]}
# data = pd.DataFrame(data).T
# plot(data)


def parse_log(log_file: str):
    precision = 2
    
    with open(log_file, "r") as f:
        log_content = f.read()

    if re.compile("OOM|Traceback").findall(log_content):
        return None

    # 每秒消耗的样本数（最后一个值）
    match_results = re.compile("global_batch_size .+ (\d+)").findall(log_content)
    global_batch_size = int(float(match_results[0])) if match_results else 1
    # match_results = re.compile("consumed samples:\ +(\d+)").findall(log_content)
    # consumed_samples =  list(map(float, match_results))[-1] if match_results else 0
    match_results =  re.compile("elapsed time per iteration \(ms\): ([\d\.]+)").findall(log_content)
    elapse_time_per_iter = list(map(float, match_results))[-1] if match_results else 1
    samples_per_second = round(float(global_batch_size / (elapse_time_per_iter / 1000)), precision)

    time_metrics = ["forward-backward", "forward-compute", 
                    "backward-compute", "optimizer",
                    "grads-all-reduce"]

    time_metric_dict = {}

    for metric in time_metrics:
        match_results = re.compile(f" {metric} .+, ([\d\.]+)").findall(log_content)
        match_results = list(map(lambda x: float(x), match_results))[1:]
        # [1:]均值
        val = sum(match_results)/len(match_results) if match_results else 0
        time_metric_dict[metric] = round(val, precision)

    # print(time_metric_dict)

    # 显存占用指标（最大值）
    match_results = re.compile("max allocated: ([\d\.]+)").findall(log_content)
    match_results = list(map(lambda x: float(x), match_results))
    memory_size = int(max(match_results)) if match_results else 0

    metrics =  Metrics(forward_backward=time_metric_dict["forward-backward"],
                    forward=time_metric_dict["forward-compute"],
                    backward=time_metric_dict["backward-compute"],
                    optimizer=time_metric_dict["optimizer"],
                    communication=time_metric_dict["grads-all-reduce"],
                    memory=memory_size,
                    samples_per_second=samples_per_second)

    return metrics

# metrics = parse_log("train.log")
# print(metrics)


def train_model(cfg: Config):
    # global_size = cfg.batch_size * cfg.gpus_per_node

    cmd = f"""torchrun --nproc_per_node {cfg.gpus_per_node} --nnodes {cfg.nnodes} --node_rank 0 pretrain_gpt.py \
    --distributed-backend nccl \
    --tensor-model-parallel-size {cfg.tensor_parallel_size} \
    --pipeline-model-parallel-size {cfg.pipeline_parallel_size} \
    --num-layers {cfg.lm.num_layers} \
    --hidden-size {cfg.lm.hidden_size} \
    --num-attention-heads {cfg.lm.num_attention_head} \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size {cfg.batch_size} \
    --train-iters {cfg.steps} \
    --lr 0.00015 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --log-interval 10 \
    --timing-log-level 2 \
    --data-path /data/datasets/gpt2/BookCorpusDataset_text_document \
    --vocab-file /data/datasets/gpt2/gpt2-vocab.json \
    --merge-file /data/datasets/gpt2/gpt2-merges.txt \
    --data-impl mmap """

    if cfg.fp16:
        cmd += " --fp16 "

    if cfg.activation_ckpt:
        cmd += " --recompute-activations "

    if cfg.grad_accum:
        cmd += " --accumulate-allreduce-grads-in-fp32 "

    # 日志输出到train.log文件
    cmd += " > train.log 2>&1"

    # print(cmd)
    print("training model ......")
    os.system(cmd)
