import torch
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from vllm import LLM

def train_func():
    llm = LLM('facebook/opt-6.7b', tensor_parallel_size=2, gpu_memory_utilization=0.8)
    output = llm.generate('San Francisco is a')
    print(output)
    trainer = TorchTrainer(train_func, scaling_config=ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={'GPU': 2}))
    trainer.fit()

train_func()
