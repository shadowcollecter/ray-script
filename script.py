# import torch
# from ray.train import ScalingConfig
# from ray.train.torch import TorchTrainer
# from vllm import LLM

# def train_func():
#     llm = LLM('facebook/opt-6.7b', tensor_parallel_size=2, gpu_memory_utilization=0.8)
#     output = llm.generate('San Francisco is a')
#     print(output)
    

# trainer = TorchTrainer(train_func, 
#                        scaling_config=ScalingConfig(
#                             num_workers=2, 
#                             use_gpu=True, 
#                             resources_per_worker={'GPU': 1}
#                             )
#                        )

# trainer.fit()



# import ray

# ray.init()

# @ray.remote(num_gpus=1)
# class GPUActor:
#     def say_hello(self):
#         print("I live in a pod with GPU access.")

# # Request actor placement.
# gpu_actors = [GPUActor.remote() for _ in range(2)]
# # The following command will block until two Ray pods with GPU access are scaled
# # up and the actors are placed.
# ray.get([actor.say_hello.remote() for actor in gpu_actors])

import ray
from vllm import LLM

ray.init()

@ray.remote(num_gpus=2)
class LLMActor:
    def __init__(self):
        self.llm = LLM('facebook/opt-6.7b', tensor_parallel_size=2, gpu_memory_utilization=0.8)
    
    def generate(self, text):
        return self.llm.generate(text)
    

# run an actor
llm_actor = LLMActor.remote()
output = ray.get(llm_actor.generate.remote('San Francisco is a'))
