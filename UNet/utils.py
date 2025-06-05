import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
import pdb
import time
import wandb

# Custom Dataset
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_names = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.image_names[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


# Transformations
class SegmentationTransform:
    def __call__(self, image, mask):
        # Resize
        image = TF.resize(image, (256, 256))
        mask = TF.resize(mask, (256, 256), interpolation=Image.NEAREST)

        # To tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask


# U-Net Model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.down1 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bridge = conv_block(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_block2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_block1 = conv_block(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        b = self.bridge(self.pool2(d2))
        u2 = self.up_block2(torch.cat([self.up2(b), d2], dim=1))
        u1 = self.up_block1(torch.cat([self.up1(u2), d1], dim=1))
        return torch.sigmoid(self.out(u1))


# Trainer
class UNetTrainer:
    def __init__(self, model, train_loader, val_loader, device, wandb_log):
        self.model = model.to(device)
        self.wandb_log = wandb_log
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)

    def load_checkpoint(self, checkpoint_path):
        if not checkpoint_path:
            print("Checkpoint is None, starting a fresh model")
        elif os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"Loaded model weights from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting fresh.")


    def train(self, num_epochs=10, checkpoint_path=None, validate_val_every=10):
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_dice = 0
            num_batches = 0

            for images, masks in self.train_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Compute Dice score
                preds = (outputs > 0.5).float()
                dice = self.dice_score(preds, masks)

                train_loss += loss.item()
                train_dice += dice.item()
                num_batches += 1

            avg_train_loss = train_loss / num_batches
            avg_train_dice = train_dice / num_batches

            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training Dice: {avg_train_dice:.4f}")

            # Log training metrics to wandb
            if self.wandb_log:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "train_dice": avg_train_dice
                })

            # Validation phase (every `validate_every` epochs)
            if (epoch + 1) % validate_val_every == 0:
                self.model.eval()
                val_loss = 0
                val_dice = 0
                num_batches = 0

                with torch.no_grad():
                    for images, masks in self.val_loader:
                        images, masks = images.to(self.device), masks.to(self.device)
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)

                        # Compute Dice score
                        preds = (outputs > 0.5).float()
                        dice = self.dice_score(preds, masks)

                        val_loss += loss.item()
                        val_dice += dice.item()
                        num_batches += 1

                avg_val_loss = val_loss / num_batches
                avg_val_dice = val_dice / num_batches

                print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Dice: {avg_val_dice:.4f}")

                # Log validation metrics to wandb
                if self.wandb_log:
                    wandb.log({
                        "epoch": epoch + 1,
                        "val_loss": avg_val_loss,
                        "val_dice": avg_val_dice
                    })

        if checkpoint_path:
            # Save model at the end of training
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Model weights saved to {checkpoint_path}")
            
    @staticmethod
    def dice_score(pred, target, epsilon=1e-6):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return (2. * intersection + epsilon) / (union + epsilon)

class Evaluator:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def evaluate(self, dataloader):
        self.model.eval()
        dice_scores = []
        accuracies = []

        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                preds = (outputs > 0.5).float()

                dice = self.dice_score(preds, masks)
                acc = (preds == masks).float().mean().item()

                dice_scores.append(dice)
                accuracies.append(acc)

        avg_dice = sum(dice_scores) / len(dice_scores)
        avg_acc = sum(accuracies) / len(accuracies)
        print(f"Validation Accuracy: {avg_acc:.4f}, Dice Score: {avg_dice:.4f}")

    @staticmethod
    def dice_score(pred, target, epsilon=1e-6):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return (2. * intersection + epsilon) / (union + epsilon)

def predict_single_image(model, image_path, transform, device):
    t0 = time.time()
    model.eval()

    # Load original image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)

    # Apply transform (resize to 256x256 and tensor conversion)
    dummy_mask = Image.new("L", image.size)
    image_tensor, _ = transform(image, dummy_mask)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = (output > 0.5).float().squeeze().cpu()  # shape: [H, W]

    # Convert predicted mask tensor to PIL Image and resize back
    pred_mask_img = TF.to_pil_image(pred_mask)  # from tensor [0,1] → PIL grayscale
    pred_mask_img = pred_mask_img.resize(original_size, resample=Image.NEAREST)

    print("Entering Debug Mode inside predict_single_image()")
    t1 = time.time()

    print(t1-t0)

    # pdb.set_trace()

    # Plot side-by-side
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask_img, cmap="gray")
    plt.title("Predicted Mask (Resized)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Optional: save prediction
    # pred_mask_img.save("prediction.png")
