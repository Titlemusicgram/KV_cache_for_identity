from torch import bfloat16


local_files_only = False
dtype = bfloat16
device_map = "auto"
MODEL_NAME = "Qwen/Qwen3-32B"

# local_files_only = True
# dtype = None
# device_map = None
# MODEL_NAME = 'Qwen/Qwen3-0.6B'

start_seed = 50
seeds_amount = 1

IM_START_TOKEN_ID = 151644
IM_END_TOKEN_ID = 151645

# MODEL_NAME = "Qwen/Qwen3-4B"
