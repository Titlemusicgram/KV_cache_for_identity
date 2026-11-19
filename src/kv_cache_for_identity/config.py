from torch import bfloat16

# local_files_only = True
# dtype = None
# device_map = None

local_files_only = False
dtype = bfloat16
device_map = "auto"

# MODEL_NAME = 'Qwen/Qwen3-0.6B'
# MODEL_NAME = "Qwen/Qwen3-4B"
MODEL_NAME = "Qwen/Qwen3-32B"

IM_START_TOKEN_ID = 151644
IM_END_TOKEN_ID = 151645
