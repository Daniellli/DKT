import os
import uuid
import time

import numpy as np 
from tqdm import tqdm 
from loguru import logger




# ResourceQueueName: "cwm-L20-8card"
# ResourceQueueName: "cwm-L20-1card"

template = """# self define e.g text_classfication
TaskName: "{task_name}"
Description: "transparent object perception"
Entrypoint: "{command}"
ImageUrl: "baai-cwm-cr01-cn-beijing.cr.volces.com/baai-cwm-cr-namespace/diffsynth:v7"

ResourceQueueName: "cwm-L20-8card"
Framework: "PyTorchDDP"
TaskRoleSpecs:
    - RoleName: "worker"
      RoleReplicas: 1
      Flavor: "ml.gni3cgd.5xlarge"
Storages:
    - Type: "Vepfs" # 可以是 Tos 或者 Vepfs.
      MountPath: "/baai-cwm-vepfs" # 在分布式训练容器中挂载的路径
      VepfsId: "vepfs-cnbj4b621a1f4a2c"
      SubPath: "/"

    - Type: "Nas" # 可以是 Tos 或者 Vepfs.
      MountPath: "/baai-cwm-backup" # 在分布式训练容器中挂载的路径
      NasId: "cnas-cnbjb0e2acbfba11e1"
      NasAddr: "cnbjb0e2acbfba11e1.mji23i10x4ow5smt1as0f0jm.nas.ivolces.com/cnas-cnbjb0e2acbfba11e1"
      


AccessType: "Public"
DelayExitTimeSeconds: "5m"
CacheType: "Cloudfs"
# user define retry options
RetryOptions:
    EnableRetry: false
    MaxRetryTimes: 5
    IntervalSeconds: 120
    PolicySets: []
# diagnosis options
DiagOptions:
    - Name: "EnvironmentalDiagnosis"
      Enable: false
    - Name: "PythonDetection"
      Enable: false
    - Name: "LogDetection"
      Enable: false"""
      



# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS-8gpus-e2e-rgb_depth-w320-h240-Wan2.1-Fun-1.3B-Control-2025-08-20-17:29:25/epoch-2-60000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS-8gpus-e2e-rgb_depth-w320-h240-Wan2.1-Fun-1.3B-Control-2025-08-20-17:29:25/epoch-1-50000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp-4gpus-origin-stage2-rgb_depth_normal-w320-h240-Wan2.1-Fun-1.3B-Control-2025-08-21-00:48:19/epoch-1-20000.safetensors'

# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS-8gpus-OG-rgb_depth-w320-h240-Wan2.1-Fun-1.3B-Control-2025-08-15-15:35:00/epoch-3-110000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS-8gpus-OG-rgb_depth-w320-h240-Wan2.1-Fun-1.3B-Control-2025-08-15-15:35:00/epoch-3-110000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS-8gpus-origin-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-23-14:43:35/epoch-0-10000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS-8gpus-origin-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-23-14:43:35/epoch-0-20000.safetensors"

# model_name = "logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS-8gpus-origin-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-23-14:43:35/epoch-1-30000.safetensors"
# model_name = "logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS-8gpus-origin-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-23-14:43:35/epoch-1-40000.safetensors"
# model_name = "logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS-8gpus-origin-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-23-14:43:35/epoch-1-50000.safetensors"
# model_name = "logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS-8gpus-origin-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-23-14:43:35/epoch-2-60000.safetensors"

#*long  
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_glassverse-4gpus-origin-stage2-sft-long-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-27-23:44:31/epoch-1-10000.safetensors"





#* lora model 
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp-4gpus-origin-stage2-rgb_normal-w320-h240-Wan2.1-Fun-1.3B-Control-2025-08-22-12:58:11/epoch-1-20000.safetensors'
# model_name='logs/outs/train/sft-T2SQNet_glassverse_t2sqnet2_glassverse-8gpus-origin-stage2-long-lora-rgb_depth-w320-h240-Wan2.1-Fun-1.3B-Control-2025-08-23-03:02:39/epoch-0.safetensors'

#* 14B depth
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS-4gpus-origin-lora-14B-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-23-16:14:48/epoch-0-10000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS-4gpus-origin-lora-14B-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-23-16:14:48/epoch-0-20000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS-4gpus-origin-lora-14B-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-23-16:14:48/epoch-0-30000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS-4gpus-origin-lora-14B-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-23-16:14:48/epoch-0-40000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS-4gpus-origin-lora-14B-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-23-16:14:48/epoch-0-50000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS-4gpus-origin-lora-14B-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-23-16:14:48/epoch-1-60000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS-4gpus-origin-lora-14B-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-23-16:14:48/epoch-1-70000.safetensors"



#* 14B depth, new
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse-8gpus-origin-lora-14B-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-09-09-00:28:49/epoch-2-86000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-8gpus-origin-lora-14B-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-28-23:27:10/epoch-0-10000.safetensors"
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-8gpus-origin-lora-14B-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-28-23:27:10/epoch-0-15000.safetensors'
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-8gpus-origin-lora-14B-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-28-23:27:10/epoch-0-20000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-8gpus-origin-lora-14B-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-28-23:27:10/epoch-0-25000.safetensors"
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-8gpus-origin-lora-14B-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-28-23:27:10/epoch-0-30000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-8gpus-origin-lora-14B-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-28-23:27:10/epoch-0-35000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-8gpus-origin-lora-14B-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-28-23:27:10/epoch-0.safetensors'




#* 1.3B depth, new

# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-1.3B-rgb_depth-w832-h480-Wan2.1-Fun-Control-2025-10-28-23:26:41/epoch-0-10000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-1.3B-rgb_depth-w832-h480-Wan2.1-Fun-Control-2025-10-28-23:26:41/epoch-0-20000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-1.3B-rgb_depth-w832-h480-Wan2.1-Fun-Control-2025-10-28-23:26:41/epoch-0-30000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-1.3B-rgb_depth-w832-h480-Wan2.1-Fun-Control-2025-10-28-23:26:41/epoch-0-40000.safetensors'
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-1.3B-rgb_depth-w832-h480-Wan2.1-Fun-Control-2025-10-28-23:26:41/epoch-0-45000.safetensors"
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-1.3B-rgb_depth-w832-h480-Wan2.1-Fun-Control-2025-10-28-23:26:41/epoch-0-50000.safetensors'




#* 14B depth, stage 2
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_glassverse-8gpus-origin-lora-14B-stage2-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-09-02-19:14:43/epoch-0-2000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_glassverse-8gpus-origin-lora-14B-stage2-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-09-02-19:14:43/epoch-1-4000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_glassverse-8gpus-origin-lora-14B-stage2-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-09-02-19:14:43/epoch-1-6000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_glassverse-8gpus-origin-lora-14B-stage2-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-09-04-15:55:26/epoch-2-8000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_glassverse-8gpus-origin-lora-14B-stage2-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-09-04-15:55:26/epoch-2-10000.safetensors"




#*normal model
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp-4gpus-origin-stage1-stf-rgb_normal-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-23-16:40:22/epoch-0-10000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp-4gpus-origin-stage1-stf-rgb_normal-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-23-16:40:22/epoch-1-20000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp-4gpus-origin-stage1-stf-rgb_normal-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-23-16:40:22/epoch-2-30000.safetensors"
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp-4gpus-origin-stage1-stf-rgb_normal-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-23-16:40:22/epoch-2-40000.safetensors'
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp-4gpus-origin-stage1-stf-rgb_normal-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-23-16:40:22/epoch-3-50000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp-4gpus-origin-stage1-stf-rgb_normal-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-23-16:40:22/epoch-4-60000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp-4gpus-origin-stage1-stf-rgb_normal-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-23-16:40:22/epoch-5-70000.safetensors"

#* 14B lora, normal 
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_glassverse-4gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-08-27-13:22:57/epoch-0-10000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_glassverse-4gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-08-27-13:22:57/epoch-1-20000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_glassverse-4gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-08-27-13:22:57/epoch-1-30000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_glassverse-4gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-08-27-13:22:57/epoch-2-40000.safetensors"
# model_name ="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_glassverse-4gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-08-27-13:22:57/epoch-2-50000.safetensors"
# model_name ="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_glassverse-4gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-08-27-13:22:57/epoch-3-60000.safetensors"
# model_name ="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_glassverse-4gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-08-27-13:22:57/epoch-4-70000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_glassverse-4gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-08-27-13:22:57/epoch-4-80000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_glassverse-4gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-08-27-13:22:57/epoch-5-90000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_glassverse-4gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-08-27-13:22:57/epoch-5-100000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_glassverse-4gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-08-27-13:22:57/epoch-6-110000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_glassverse-4gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-08-27-13:22:57/epoch-6-120000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_glassverse-4gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-08-27-13:22:57/epoch-7-130000.safetensors"

# model_name='logs/outs/train/sft-T2SQNet_glassverse_cleargrasp_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-09-23-17:41:45/epoch-0-10000.safetensors'
# model_name='logs/outs/train/sft-T2SQNet_glassverse_cleargrasp_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-09-23-17:41:45/epoch-0-20000.safetensors'
# model_name='logs/outs/train/sft-T2SQNet_glassverse_cleargrasp_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-09-23-17:41:45/epoch-1-30000.safetensors'
# model_name='logs/outs/train/sft-T2SQNet_glassverse_cleargrasp_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-09-23-17:41:45/epoch-1-40000.safetensors'
# model_name="logs/outs/train/sft-T2SQNet_glassverse_cleargrasp_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-09-23-17:41:45/epoch-2-50000.safetensors"
# model_name="logs/outs/train/sft-T2SQNet_glassverse_cleargrasp_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-09-23-17:41:45/epoch-2-60000.safetensors"
# model_name="logs/outs/train/sft-T2SQNet_glassverse_cleargrasp_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-09-23-17:41:45/epoch-3-70000.safetensors"
# model_name="logs/outs/train/sft-T2SQNet_glassverse_cleargrasp_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-09-23-17:41:45/epoch-3-80000.safetensors"


# model_name='logs/outs/train/sft-T2SQNet_glassverse_cleargrasp_interiorverse_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-09-28-14:38:14/epoch-0-5000.safetensors'
# model_name='logs/outs/train/sft-T2SQNet_glassverse_cleargrasp_interiorverse_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-09-28-14:38:14/epoch-0-10000.safetensors'
# model_name='logs/outs/train/sft-T2SQNet_glassverse_cleargrasp_interiorverse_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-09-28-14:38:14/epoch-0-20000.safetensors'
# model_name='logs/outs/train/sft-T2SQNet_glassverse_cleargrasp_interiorverse_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-09-28-14:38:14/epoch-1-30000.safetensors'
# model_name='logs/outs/train/sft-T2SQNet_glassverse_cleargrasp_interiorverse_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-09-28-14:38:14/epoch-1-40000.safetensors'
# model_name="logs/outs/train/sft-T2SQNet_glassverse_cleargrasp_interiorverse_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-09-28-14:38:14/epoch-1-50000.safetensors"
# model_name="logs/outs/train/sft-T2SQNet_glassverse_cleargrasp_interiorverse_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-09-28-14:38:14/epoch-2-60000.safetensors"
# model_name="logs/outs/train/sft-T2SQNet_glassverse_cleargrasp_interiorverse_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-14B-rgb_normal-w832-h480-Wan2.1-Fun-Control-2025-09-28-14:38:14/epoch-2-70000.safetensors"



# mask_model
# model_name="logs/outs/train/sft-T2SQNet_Trans10K_cleargrasp-8gpus-origin-stage1-lora-14B-rgb_mask-w832-h480-Wan2.1-Fun-Control-2025-09-25-00:40:25/epoch-15-95000.safetensors"
# model_name="logs/outs/train/sft-T2SQNet_Trans10K_cleargrasp-8gpus-origin-stage1-lora-14B-rgb_mask-w832-h480-Wan2.1-Fun-Control-2025-09-25-00:40:25/epoch-14-90000.safetensors"
# model_name="logs/outs/train/sft-T2SQNet_Trans10K_cleargrasp-8gpus-origin-stage1-lora-14B-rgb_mask-w832-h480-Wan2.1-Fun-Control-2025-09-25-00:40:25/epoch-12-80000.safetensors"
# model_name="logs/outs/train/sft-T2SQNet_Trans10K_cleargrasp-8gpus-origin-stage1-lora-14B-rgb_mask-w832-h480-Wan2.1-Fun-Control-2025-09-25-00:40:25/epoch-11-70000.safetensors"
# model_name="logs/outs/train/sft-T2SQNet_Trans10K_cleargrasp-8gpus-origin-stage1-lora-14B-rgb_mask-w832-h480-Wan2.1-Fun-Control-2025-09-25-00:40:25/epoch-9-60000.safetensors"

# model_name="logs/outs/train/sft-T2SQNet_Trans10K_cleargrasp-8gpus-origin-stage1-lora-14B-rgb_mask-w832-h480-Wan2.1-Fun-Control-2025-09-25-00:40:25/epoch-7-50000.safetensors"
# model_name="logs/outs/train/sft-T2SQNet_Trans10K_cleargrasp-8gpus-origin-stage1-lora-14B-rgb_mask-w832-h480-Wan2.1-Fun-Control-2025-09-25-00:40:25/epoch-6-40000.safetensors"
# model_name="logs/outs/train/sft-T2SQNet_Trans10K_cleargrasp-8gpus-origin-stage1-lora-14B-rgb_mask-w832-h480-Wan2.1-Fun-Control-2025-09-25-00:40:25/epoch-4-30000.safetensors"
# model_name="logs/outs/train/sft-T2SQNet_Trans10K_cleargrasp-8gpus-origin-stage1-lora-14B-rgb_mask-w832-h480-Wan2.1-Fun-Control-2025-09-25-00:40:25/epoch-3-20000.safetensors"
# model_name="logs/outs/train/sft-T2SQNet_Trans10K_cleargrasp-8gpus-origin-stage1-lora-14B-rgb_mask-w832-h480-Wan2.1-Fun-Control-2025-09-25-00:40:25/epoch-1-10000.safetensors"


#* stage2 LoRA model
# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora-14B-stage2-rgb_depth_mask-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-17-09:55:14/epoch-0-10000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora-14B-stage2-rgb_depth_mask-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-17-09:55:14/epoch-1-20000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora-14B-stage2-rgb_depth_mask-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-17-09:55:14/epoch-2-30000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora-14B-stage2-rgb_depth_mask-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-17-09:55:14/epoch-3-40000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora-14B-stage2-rgb_depth_mask-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-17-09:55:14/epoch-4-50000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora-14B-stage2-rgb_depth_mask-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-17-09:55:14/epoch-5-60000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora-14B-stage2-rgb_depth_mask-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-17-09:55:14/epoch-6-70000.safetensors'




#* all parameters

# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora16-14B-stage2-rgb_depth_mask-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-21-13:50:01/epoch-0-10000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora16-14B-stage2-rgb_depth_mask-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-21-13:50:01/epoch-1-20000.safetensors'
# model_name="logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora16-14B-stage2-rgb_depth_mask-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-21-13:50:01/epoch-2-30000.safetensors"




#* mask, self-attn

# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora64-14B-rgb_depth_mask-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-29-22:28:19/epoch-0-10000.safetensors'

# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora64-14B-rgb_depth_mask-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-29-22:28:19/epoch-1-15000.safetensors'

# model_name="logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora64-14B-rgb_depth_mask-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-29-22:28:19/epoch-1-20000.safetensors"
# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora64-14B-rgb_depth_mask-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-29-22:28:19/epoch-2-30000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora64-14B-rgb_depth_mask-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-29-22:28:19/epoch-3-40000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora64-14B-rgb_depth_mask-w832-h480-Wan2.1-Fun-1.3B-Control-2025-10-29-22:28:19/epoch-4-50000.safetensors'



#* mask ,128rank,  self-attn and ffn
# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora128-14B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-03-10:27:53/epoch-0-5000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora128-14B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-03-10:27:53/epoch-0-10000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora128-14B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-03-10:27:53/epoch-1-15000.safetensors'

# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora128-14B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-03-10:27:53/epoch-1-20000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora128-14B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-03-10:27:53/epoch-2-30000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora128-14B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-03-10:27:53/epoch-3-40000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora128-14B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-03-10:27:53/epoch-4-50000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_cleargrasp-4gpus-origin-lora128-14B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-03-10:27:53/epoch-4-55000.safetensors'




#* 1.3B lora
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-27-01:21:28/epoch-0-10000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-27-01:21:28/epoch-0-20000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-27-01:21:28/epoch-0-30000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-27-01:21:28/epoch-1-40000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-27-01:21:28/epoch-1-50000.safetensors"
# model_name = "logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-27-01:21:28/epoch-1-60000.safetensors"
# model_name = "logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-27-01:21:28/epoch-2-70000.safetensors"
# model_name ="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-27-01:21:28/epoch-2-80000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-27-01:21:28/epoch-2-90000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-27-01:21:28/epoch-3-100000.safetensors"
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse-8gpus-origin-stage1-lora-rgb_depth-w832-h480-Wan2.1-Fun-1.3B-Control-2025-08-27-01:21:28/epoch-3-110000.safetensors"




#* acc gradient 
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-8accu-1.3B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-07-21:42:24/epoch-0-10000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-8accu-1.3B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-07-21:42:24/epoch-0-20000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-8accu-1.3B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-07-21:42:24/epoch-0-30000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-8accu-1.3B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-07-21:42:24/epoch-0-40000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-8accu-1.3B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-07-21:42:24/epoch-0-50000.safetensors'

# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-8accu-1.3B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-07-21:42:24/epoch-0-60000.safetensors'
# model_name="logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-8accu-1.3B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-07-21:42:24/epoch-0-70000.safetensors"

#* joint training mask 
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-1.3B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-07-18:31:20/epoch-0-10000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-1.3B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-07-18:31:20/epoch-0-20000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-1.3B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-07-18:31:20/epoch-0-30000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-1.3B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-07-18:31:20/epoch-0-40000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-1.3B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-07-18:31:20/epoch-0-50000.safetensors'

# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-1.3B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-07-18:31:20/epoch-0-60000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-1.3B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-07-18:31:20/epoch-0-70000.safetensors'


#* mask 

# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-1.3B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-12-16:18:36/epoch-0-10000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-1.3B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-12-16:18:36/epoch-0-20000.safetensors'
# model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-1.3B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-12-16:18:36/epoch-0-30000.safetensors'
model_name='logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-1.3B-rgb_depth_mask-w832-h480-Wan2.1-Fun-Control-2025-11-12-16:18:36/epoch-0-40000.safetensors'



rm_str="/".join(model_name.split("/")[:-1])
# logger.info(f"rm_str: {rm_str}")

training_strategy='origin'



data_jsonl_path=[
  
    'data/clearpose/test.jsonl', 
    # 'data/DREDS/test_std_catnovel.jsonl', 
    # 'data/DREDS/test_std_catknown-test.jsonl',     
    # "data/rendering_testset/test_final.jsonl"

    # "data/pics/pics.jsonl",
    # 'data/housecat6d/test_raw_depth.jsonl',
    # 'data/TRansPose/test_uniform_1_with_video.jsonl',
    #  'data/phocal/test_all.jsonl',
    # 'data/TransCG/test.jsonl',
    # 'data/Trans10K_cls12/mini_test_005.jsonl', 
    # 'data/Trans10K_cls12/mini_test_200.jsonl', 

      ]
      
window_size=21
overlap=5

inference_steps=5
commands = []

#* task switcher
# prompt='normal'
prompt='depth'
# prompt='transparent'


is_lora=True
# is_lora=False
# is_14b=True
is_14b=False


new_lora_weight=1.0

for data_path in data_jsonl_path:

  if 'phocal' in data_path.lower() or 'transcg' in data_path.lower() or 'housecat' in data_path.lower() or 'dreds' in data_path.lower():
    infer_length = 10000
  elif 'transpose' in data_path.lower():
    infer_length=81
  else:
    infer_length=301

    
    
  commond1 = f"bash /baai-cwm-vepfs/cwm/shaocong.xu/exp/DiffSynth-Studio/infer-eval.sh \
    {model_name} {data_path} 0 {window_size} {overlap} {infer_length} {training_strategy} {inference_steps} {prompt} {is_lora} {is_14b} {new_lora_weight}"
  commands.append(commond1)




# commands = [' '.join(x.split(" ")[1:]) for x in command.split('\n') if x]
taskname_prefix = "daniel_"
# taskname is random string with length 8
taskname_randoms = [
  # replace the command. only keep a-zA-Z0-9
]
for command in commands:
    new_random = ""
    for char in command.replace("/baai-cwm-vepfs/cwm/shaocong.xu/exp/DiffSynth-Studio", "").replace(rm_str, "").lower():
        if char.isalnum():
            new_random += char
        else:
            new_random += "-"
    taskname_randoms.append(new_random)
tasknames = [f'{taskname_prefix}_{random}' for random in taskname_randoms]





print(len(tasknames), tasknames[0])

for taskname, command in zip(tasknames, commands):
    logger.warning(f"taskname: {taskname}, command: {command}")
    
    
    with open(f'scripts/task_yaml/{taskname}.yaml', 'w') as f:
        f.write(template.format(task_name=taskname, command=command))
    print(f'volc ml_task submit --conf ./scripts/task_yaml/{taskname}.yaml /// {command}')
    # time.sleep(3)
    os.system(f'volc ml_task submit --conf ./scripts/task_yaml/{taskname}.yaml')
    # remove config file
    os.system(f'rm ./scripts/task_yaml/{taskname}.yaml')
    




# time.sleep(1)
# for taskname in tasknames:
#     # use `volc ml_task submit --conf ./tf_custom_mnist_random_1worker-2c4g.yaml` to submit task
#     os.system(f'volc ml_task submit --conf ./scripts/task_yaml/{taskname}.yaml')
