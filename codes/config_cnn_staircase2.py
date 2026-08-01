#  Copyright (c) 2024 CGLab, GIST. All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without modification, 
#  are permitted provided that the following conditions are met:
#  
#  - Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation 
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors 
#    may be used to endorse or promote products derived from this software 
#    without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch

config = {}
# ======================================
# Global Variables
# =======================================
config["timezone"] = None
config["num_GPU"] = torch.cuda.device_count()
print("Num of GPU: {}".format(config["num_GPU"]))
if config["num_GPU"] > 1:
    config["multi_gpu"]  = True
else:
    config["multi_gpu"]  = False

# for test using pre-trained checkpoint
config["multi_gpu"]  = False    

config["dataloader_numworker"] = 2

# =======================================
# Dataset
# =======================================
config["DataDirectory"] = "/workspace/data"
config["trainDatasetDirectory"] = "/workspace/data/__train_scenes__/staircase2_overfit_jsa"

config["train_input"] = 'input'
config["train_target"] = 'target'


# =======================================
# Dataset
# =======================================
config["testDatasetDirectory"] = "/workspace/data/__test_scenes__/staircase2_overfit_jsa"

config["test_input"] = 'input' 
config["test_target"] = 'target'

# =======================================
# Task
# =======================================
config["task"] = 'jsaCNN_staircase2_overfit' # folder name to contain training results
# config["task"] = 'jsa_trained' # folder name to contain training results

# =======================================
# Model
# =======================================
config["embed_dim"] = 32                    # dim of emdeding features
config["x_dim"] = 3 
config["f_dim"] = 7
            # color =                      3 ch
            # aux_features = 0:2 - albedo  3 ch                           
                            # 3:5 - normal 3 ch
                            # 6 - depth    1 ch
                            

# =======================================
# Training parameter
# =======================================
config["data_dir"] = "/workspace/data"
config["manual_seed"] = 1234

config["optimizer"] = "adamw"               # optimizer for training
config["lr_initial"] = 0.0002 # 2e-4        # initial learning rate 
config["weight_decay"] = 0.001               # weight decay
config["wramup"] = True
config["warmup_epochs"] = 5

config["patch_size"] = 128
config["patch_stride_size"] = config["patch_size"]   # default
config["patch_based_learning"] = True
config["batch_size"] = 1                   # adjust to GPU memory size

# tile size for sliding-window eval (must be a multiple of 128; see eval.tiled_forward)
config["eval_tile_size"] = config["patch_size"]
# overlap (px) between neighboring eval tiles, cosine-blended to hide seams
config["eval_tile_overlap"] = 24

config["shuffle_file_list"] = True
config["dataloader_shuffle"] = True
config["aug_mode"] = False # to see overfitting, set False

config["max_epoch"] = 100                  # the max epoch for training
config["snapshot"] = 20

# config["retrain"] = True
config["retrain"] = False
config["restore_epoch"] = 50

# =======================================
# Test parameter
# =======================================
config["load_epoch"] = "best" #'jsa_pretrained'













