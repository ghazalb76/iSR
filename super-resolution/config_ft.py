# Global configuration:
seed = 42

# Model configuation:
feature_channels = 96
num_blocks = 4
act_type = "gelu"
is_train = True
model_name = "RT4KSR_Rep"

# Data configuration:
dataroot = "../Datasets/GameFrames"
scale = 8
batch_size = 1
num_workers = 4
crop_size = 256
image_format = "png"
preupsample = False
prefetch_factor = 16
rgb_range = 1.0

# Checkpoint configuration:
save_top_k = 10
checkpoint_root = "checkpoints"
# 
# Logging configuration (Tensorboard):
logger_save_dir = "logs"
logger_name = "RT4KSR_Rep"

# Optimizer configuration:
optimizer = "AdamW"     # ["AdamW", "Adam", "SGD"]

# MultiStepLR configuration:
multistepLR_milestones = [50]
multistepLR_gamma = 0.5

# lr monitor configuration:
lr_monitor_logging_interval="epoch"

# early stopping configuration:
early_stopping_patience = 50

# Training configuration:
learning_rate = 1e-4
max_epochs = 6000
accelerator = "auto"
device = "auto"
continue_training = False
checkpoint_path_continue = "checkpoints/last.ckpt"
# Eval configuration:
eval_reparameterize = True
checkpoint_path_eval = "checkpoints/best.ckpt"
eval_lr_image_dir = "dataset/test/scale4"
eval_hr_image_dir = "dataset/test/scale4/SR"
val_save_path = "results/val/Set5"

# Inference configuration:
infer_reparameterize = True
checkpoint_path_infer= "checkpoints/BeatSaber/scale8Channels96Block4.ckpt"



infer_lr_image_path = "/localhome/gba50/Finalize/dataset/test/cr/resultsOfRTX4000/scale10/stitched"

infer_save_path = "/localhome/gba50/Finalize/dataset/test/cr/resultsOfRTX4000/scale10/stitched/SR"
ground_truth_path = "/localhome/gba50/Finalize/dataset/hr/test/"

# Video inference configuration:
video_infer_reparameterize = True
checkpoint_path_video_infer = "checkpoints/best.ckpt"
video_infer_video_path = "examples/Rainforest_360.mp4"
video_infer_save_path = "results/video"
video_format = ".mp4"

# Application configuration:
app_reparameterize = True
checkpoint_path_app = "checkpoints/best.ckpt"