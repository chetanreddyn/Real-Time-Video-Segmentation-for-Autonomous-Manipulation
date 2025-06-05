import os
import pdb

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
import wandb

from utils import *


training_config = {
        "num_epochs": 10,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "batch_size": 4
    }

# wandb.init(
#     project="video-segmentation-unet",
#     name="unet_run_1",  # Change this for each run if desired
#     config=training_config
# )

# project_folder = "/home/chetan/Desktop/Acads/CS231n/Project/Video-Segmentation-for-Autonomous-Manipulation/"
project_folder = "/Users/chetan/Desktop/Spring 2025/CS231n/Project/Video-Segmentation-for-Autonomous-Manipulation/"
save_checkpoint_path = os.path.join(project_folder, "UNet", "Saved Models", "test.pth")
# load_checkpoint_path = None
load_checkpoint_path = os.path.join(project_folder, "UNet", "Saved Models", "test.pth")
num_epochs = 10
image_dir = os.path.join(project_folder, "Data/Three Handed Task/raw_images")
mask_dir = os.path.join(project_folder, "Data/Three Handed Task/binary_masks")

assert os.path.exists(image_dir), f"image_dir: {image_dir}"
assert os.path.exists(mask_dir), f"mask_dir: {mask_dir}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = SegmentationTransform()
dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)

# pdb.set_trace()
# Optional: use only a subset of 400 frames at intervals of 100
target_freq = 10
present_freq = 30
frame_skip_size = present_freq//target_freq
subset_indices = list(range(0, len(dataset), frame_skip_size))
dataset = Subset(dataset, subset_indices)

# === Train/Test split ===
# train_test_ratio = 0.8
# train_size = int(train_test_ratio * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataset = Subset(dataset, list(range(0,100)))
test_dataset = Subset(dataset, list(range(100,150)))

# DataLoader for training
train_dataloader = DataLoader(train_dataset, batch_size=training_config["batch_size"], shuffle=True)

# Optional: DataLoader for testing (if needed)
test_dataloader = DataLoader(test_dataset, batch_size=training_config["batch_size"], shuffle=False)

# pdb.set_trace()

model = UNet().to(device)
trainer = UNetTrainer(model, train_dataloader, device)
trainer.load_checkpoint(load_checkpoint_path)
# trainer.train(num_epochs=training_config["num_epochs"], checkpoint_path=save_checkpoint_path)

# evaluator = Evaluator(model, device)
# evaluator.evaluate(train_dataloader)
# evaluator.evaluate(test_dataloader)

sample_image_name = "0000.jpg"
sample_image_path = os.path.join(image_dir, sample_image_name) 
predict_single_image(model, sample_image_path, transform, device)
