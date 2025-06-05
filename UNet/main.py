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
import os
import torch
import wandb
from torch.utils.data import DataLoader, Subset
from utils import UNet, SegmentationTransform, SegmentationDataset, UNetTrainer, Evaluator, predict_single_image


class UNetPipeline:
    def __init__(self, project_folder, training_config):
        self.project_folder = project_folder
        self.training_config = training_config
        self.dataset_folder = dataset_folder
        self.save_checkpoint_name = training_config["save_checkpoint_name"]
        self.load_checkpoint_name = training_config["load_checkpoint_name"]
        self.wandb_log = training_config["wandb_log"]
        self.run_name = training_config["run_name"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_paths()
        if self.wandb_log:
            self._initialize_wandb()
        self._setup_dataset()
        self._setup_dataloaders()
        self.model = UNet().to(self.device)
        self.trainer = UNetTrainer(self.model, self.train_dataloader, self.test_dataloader, self.device, self.wandb_log)

    def _setup_paths(self):
        self.save_checkpoint_path = os.path.join(self.project_folder, "UNet", "Saved Models", self.save_checkpoint_name)
        if self.load_checkpoint_name:
            self.load_checkpoint_path = os.path.join(self.project_folder, "UNet", "Saved Models", self.load_checkpoint_name)
        else:
            self.load_checkpoint_path = None  
        self.image_dir = os.path.join(self.project_folder, self.dataset_folder, "raw_images")
        self.mask_dir = os.path.join(self.project_folder, self.dataset_folder, "binary_masks")

        assert os.path.exists(self.image_dir), f"image_dir: {self.image_dir} does not exist."
        assert os.path.exists(self.mask_dir), f"mask_dir: {self.mask_dir} does not exist."

    def _initialize_wandb(self):
        wandb.init(
            project="CS231n UNet",
            name=self.run_name,  # Change this for each run if desired
            config=self.training_config
        )

    def _setup_dataset(self):
        transform = SegmentationTransform()
        dataset = SegmentationDataset(self.image_dir, self.mask_dir, transform=transform)

        # Optional: Use only a subset of frames at intervals
        target_freq = 10
        present_freq = 30
        frame_skip_size = present_freq // target_freq
        subset_indices = list(range(0, len(dataset), frame_skip_size))
        dataset = Subset(dataset, subset_indices)

        # Train/Test split
        self.train_dataset = Subset(dataset, list(range(0, 150)))
        self.test_dataset = Subset(dataset, list(range(150, 200)))

    def _setup_dataloaders(self):
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.training_config["batch_size"], shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.training_config["batch_size"], shuffle=False)

    def train(self):
        print("Num Training Samples", len(self.train_dataloader))
        print("Num Val Samples", len(self.test_dataloader))
        self.trainer.load_checkpoint(self.load_checkpoint_path)
        self.trainer.train(num_epochs=self.training_config["num_epochs"], checkpoint_path=self.save_checkpoint_path)

    def evaluate(self):
        evaluator = Evaluator(self.model, self.device)
        evaluator.evaluate(self.train_dataloader)
        evaluator.evaluate(self.test_dataloader)

    def predict_sample_image(self, sample_image_name):
        sample_image_path = os.path.join(self.image_dir, sample_image_name)
        predict_single_image(self.model, sample_image_path, SegmentationTransform(), self.device)


# === Main Execution ===
if __name__ == "__main__":
    project_folder = "/home/chetan/Desktop/Acads/CS231n/Project/Video-Segmentation-for-Autonomous-Manipulation/"
    dataset_folder = "Data/Three Handed Task/"
    training_config = {
        "num_epochs": 100,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "save_checkpoint_name": "test.pth",
        "load_checkpoint_name": None,
        "wandb_log": True,
        "dataset_folder": "Data/Three Handed Task",
        "run_name": "Test"
    }

    pipeline = UNetPipeline(project_folder, training_config)

    # Uncomment the following lines to execute specific tasks
    pipeline.train()
    # pipeline.evaluate()
    pipeline.predict_sample_image("0000.jpg")