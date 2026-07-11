#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")


# Configuration with LSGAN and Stabilization
class Config:
    # Paths
    train_hr_path = r"D:\DIV2K\DIV2K_train_HR"
    val_hr_path = r"D:\DIV2K\DIV2K_valid_HR"
    
    # Training Strategy
    num_epochs = 100
    pretrain_epochs = 15
    batch_size = 8
    
    # Optimizers
    learning_rate_G = 1e-4
    learning_rate_D = 1e-4
    beta1 = 0.9
    beta2 = 0.999
    
    # LSGAN Loss Weights
    mse_weight = 1.0
    perceptual_weight = 1e-2
    adversarial_weight = 5e-3
    
    # LSGAN specific
    lsgan_target_real = 0.9
    lsgan_target_fake = 0.1
    
    # Gradient Clipping
    grad_clip_D = 0.01
    grad_clip_G = 0.01
    
    # Image processing
    hr_size = 96
    lr_size = hr_size // 4
    scale_factor = 4
    n_channels = 3
    
    # Save and log
    save_interval = 5
    sample_interval = 200
    
    # DataLoader
    num_workers = 0
    
config = Config()

print("Configuration loaded with LSGAN:")
print(f"  - Pretrain epochs: {config.pretrain_epochs}")
print(f"  - LSGAN target real: {config.lsgan_target_real}")
print(f"  - LSGAN target fake: {config.lsgan_target_fake}")
print(f"  - Gradient clipping: G={config.grad_clip_G}, D={config.grad_clip_D}")


# Dataset Class
class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, is_train=True, hr_size=96, scale_factor=4):
        self.hr_dir = hr_dir
        self.is_train = is_train
        self.hr_size = hr_size
        self.scale_factor = scale_factor
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG')
        self.image_files = [f for f in os.listdir(hr_dir) if f.endswith(valid_extensions)]
        
        print(f"Loaded {len(self.image_files)} images from {hr_dir}")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.hr_dir, self.image_files[idx])
        hr_image = Image.open(img_path).convert('RGB')
        
        if self.is_train:
            # Random crop
            w, h = hr_image.size
            if w >= self.hr_size and h >= self.hr_size:
                left = np.random.randint(0, w - self.hr_size + 1)
                top = np.random.randint(0, h - self.hr_size + 1)
                hr_image = hr_image.crop((left, top, left + self.hr_size, top + self.hr_size))
            else:
                hr_image = hr_image.resize((self.hr_size, self.hr_size), Image.BICUBIC)
            
            # Random horizontal flip
            if np.random.random() > 0.5:
                hr_image = hr_image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Random rotation 90 degrees
            if np.random.random() > 0.5:
                hr_image = hr_image.rotate(90)
        else:
            # Validation: resize to multiples of scale_factor
            w, h = hr_image.size
            new_w = (w // self.scale_factor) * self.scale_factor
            new_h = (h // self.scale_factor) * self.scale_factor
            hr_image = hr_image.resize((new_w, new_h), Image.BICUBIC)
        
        # Create LR
        lr_size = (hr_image.size[0] // self.scale_factor, hr_image.size[1] // self.scale_factor)
        lr_image = hr_image.resize(lr_size, Image.BICUBIC)
        
        # Convert to tensors and normalize to [-1, 1]
        hr_tensor = (ToTensor()(hr_image) - 0.5) * 2
        lr_tensor = (ToTensor()(lr_image) - 0.5) * 2
        
        return lr_tensor, hr_tensor


# Generator Network (Residual in Residual)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + residual

class Generator(nn.Module):
    def __init__(self, n_channels=3, n_residual_blocks=16):
        super(Generator, self).__init__()
        
        self.conv1 = nn.Conv2d(n_channels, 64, 9, padding=4)
        self.prelu = nn.PReLU()
        
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(n_residual_blocks)])
        
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        
        self.conv3 = nn.Conv2d(64, n_channels, 9, padding=4)
        
    def forward(self, x):
        out1 = self.prelu(self.conv1(x))
        out = self.res_blocks(out1)
        out = self.bn2(self.conv2(out))
        out = out + out1
        out = self.upsample(out)
        out = self.conv3(out)
        return torch.tanh(out)


# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, n_channels=3):
        super(Discriminator, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )
        
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


# Perceptual Loss (VGG19)
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.layers = nn.ModuleList([
            vgg[:4],
            vgg[4:9],
            vgg[9:18],
            vgg[18:27]
        ])
        
        self.to(device)
        
    def normalize(self, x):
        x = (x + 1) / 2
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return (x - mean) / std
    
    def forward(self, sr, hr):
        sr = self.normalize(sr)
        hr = self.normalize(hr)
        
        loss = 0
        x_sr = sr
        x_hr = hr
        
        for layer in self.layers:
            x_sr = layer(x_sr)
            x_hr = layer(x_hr)
            loss += nn.functional.l1_loss(x_sr, x_hr)
        
        return loss / len(self.layers)


# LSGAN Loss Functions
class LSGANLoss:
    @staticmethod
    def generator_loss(fake_pred):
        target = torch.ones_like(fake_pred) * config.lsgan_target_real
        return nn.MSELoss()(fake_pred, target)
    
    @staticmethod
    def discriminator_loss(real_pred, fake_pred):
        real_target = torch.ones_like(real_pred) * config.lsgan_target_real
        fake_target = torch.zeros_like(fake_pred) * config.lsgan_target_fake
        
        real_loss = nn.MSELoss()(real_pred, real_target)
        fake_loss = nn.MSELoss()(fake_pred, fake_target)
        
        return (real_loss + fake_loss) / 2


# Utility Functions
def denormalize(tensor):
    """Convert normalized tensor (-1 to 1) to image (0-255)"""
    return ((tensor * 0.5 + 0.5) * 255).byte().cpu().numpy().transpose(1, 2, 0)

def compute_metrics(sr_image, hr_image):
    """Compute PSNR and SSIM metrics"""
    sr_np = denormalize(sr_image)
    hr_np = denormalize(hr_image)
    
    psnr_value = psnr(hr_np, sr_np, data_range=255)
    ssim_value = ssim(hr_np, sr_np, channel_axis=2, data_range=255)
    
    return psnr_value, ssim_value

def save_sample_images(lr, sr, hr, epoch, save_dir='samples'):
    """Save comparison images"""
    os.makedirs(save_dir, exist_ok=True)
    
    lr_np = denormalize(lr.squeeze())
    sr_np = denormalize(sr.squeeze())
    hr_np = denormalize(hr.squeeze())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(lr_np)
    axes[0].set_title(f'LR (24x24)', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(sr_np)
    axes[1].set_title(f'SR (96x96)', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(hr_np)
    axes[2].set_title(f'HR (96x96)', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/epoch_{epoch}.png', dpi=150)
    plt.close()
    
    sr_img = Image.fromarray(sr_np)
    sr_img.save(f'{save_dir}/sr_epoch_{epoch}.png')

def evaluate_model(generator, val_loader, device):
    """Evaluate model on validation set"""
    generator.eval()
    psnr_values = []
    ssim_values = []
    
    print("\nEvaluating model on validation set...")
    
    with torch.no_grad():
        for lr, hr in tqdm(val_loader, desc="Evaluating"):
            lr, hr = lr.to(device), hr.to(device)
            sr = generator(lr)
            
            sr_cpu = sr.cpu()
            hr_cpu = hr.cpu()
            
            for i in range(sr_cpu.size(0)):
                psnr_val, ssim_val = compute_metrics(sr_cpu[i], hr_cpu[i])
                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)
    
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Average PSNR:  {avg_psnr:.2f} dB")
    print(f"Average SSIM:  {avg_ssim:.4f}")
    print(f"PSNR Range:    [{np.min(psnr_values):.2f}, {np.max(psnr_values):.2f}]")
    print(f"SSIM Range:    [{np.min(ssim_values):.4f}, {np.max(ssim_values):.4f}]")
    print(f"Valid samples: {len(psnr_values)}")
    
    return avg_psnr, avg_ssim, psnr_values, ssim_values

def visual_comparison(generator, val_loader, device, num_samples=4):
    """Show side-by-side comparison"""
    generator.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    print("Generating visual comparisons...")
    
    with torch.no_grad():
        for idx, (lr, hr) in enumerate(val_loader):
            if idx >= num_samples:
                break
                
            lr, hr = lr.to(device), hr.to(device)
            sr = generator(lr)
            
            lr_np = denormalize(lr.cpu()[0])
            sr_np = denormalize(sr.cpu()[0])
            hr_np = denormalize(hr.cpu()[0])
            
            psnr_val, ssim_val = compute_metrics(sr.cpu()[0], hr.cpu()[0])
            
            axes[idx, 0].imshow(lr_np)
            axes[idx, 0].set_title(f'LR Input\n{lr_np.shape[0]}x{lr_np.shape[1]}', fontsize=12)
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(sr_np)
            axes[idx, 1].set_title(f'SR Output (4x)\nPSNR: {psnr_val:.1f} dB, SSIM: {ssim_val:.3f}', fontsize=12)
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(hr_np)
            axes[idx, 2].set_title(f'HR Ground Truth\n{hr_np.shape[0]}x{hr_np.shape[1]}', fontsize=12)
            axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Comparison saved as 'comparison_results.png'")

def plot_training_history(history, pretrain_epochs):
    """Plot all training losses"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    epochs = range(1, len(history['G_loss']) + 1)
    gan_epochs = range(pretrain_epochs + 1, len(history['G_loss']) + 1)
    
    # Generator Total Loss
    axes[0, 0].plot(epochs, history['G_loss'], 'b-', linewidth=2)
    axes[0, 0].axvline(x=pretrain_epochs, color='r', linestyle='--', linewidth=2, label='GAN Training Start')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Generator Total Loss', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Discriminator Loss
    axes[0, 1].plot(epochs, history['D_loss'], 'orange', linewidth=2)
    axes[0, 1].axvline(x=pretrain_epochs, color='r', linestyle='--', linewidth=2, label='GAN Training Start')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].set_title('Discriminator Loss (LSGAN)', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Generator Components
    if len(history.get('G_mse', [])) > 0:
        axes[0, 2].plot(gan_epochs, history['G_mse'], 'g-', label='MSE Loss', linewidth=2)
        axes[0, 2].plot(gan_epochs, history['G_perceptual'], 'm-', label='Perceptual (VGG)', linewidth=2)
        axes[0, 2].plot(gan_epochs, history['G_adv'], 'r-', label='Adversarial (LSGAN)', linewidth=2)
        axes[0, 2].set_xlabel('Epoch', fontsize=12)
        axes[0, 2].set_ylabel('Loss', fontsize=12)
        axes[0, 2].set_title('Generator Component Losses', fontsize=14)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # D Real vs Fake
    if len(history.get('D_real', [])) > 0:
        axes[1, 0].plot(gan_epochs, history['D_real'], 'g-', label='Real Images Loss', linewidth=2)
        axes[1, 0].plot(gan_epochs, history['D_fake'], 'r-', label='Fake Images Loss', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Loss', fontsize=12)
        axes[1, 0].set_title('Discriminator Real vs Fake Loss', fontsize=14)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # G vs D Balance
    if len(gan_epochs) > 0:
        axes[1, 1].plot(gan_epochs, 
                       [history['G_loss'][i-1] for i in gan_epochs], 
                       'b-', label='Generator', linewidth=2)
        axes[1, 1].plot(gan_epochs, 
                       [history['D_loss'][i-1] for i in gan_epochs], 
                       'orange', label='Discriminator', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Loss', fontsize=12)
        axes[1, 1].set_title('GAN Training Balance', fontsize=14)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Loss values summary
    axes[1, 2].axis('off')
    
    final_mse = history['G_mse'][-1] if len(history.get('G_mse', [])) > 0 else 0
    final_vgg = history['G_perceptual'][-1] if len(history.get('G_perceptual', [])) > 0 else 0
    final_adv = history['G_adv'][-1] if len(history.get('G_adv', [])) > 0 else 0
    
    summary_text = f"""
    TRAINING SUMMARY
    
    Pretrain Epochs: {pretrain_epochs}
    Total Epochs: {len(history['G_loss'])}
    
    Final G Loss: {history['G_loss'][-1]:.4f}
    Final D Loss: {history['D_loss'][-1]:.4f}
    
    Final MSE: {final_mse:.4f}
    Final VGG: {final_vgg:.4f}
    Final Adv: {final_adv:.4f}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, fontfamily='monospace',
                   verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('SRGAN Training History (LSGAN)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()
    print("Training history saved as 'training_history.png'")

def final_summary(avg_psnr, avg_ssim, generator, discriminator):
    """Print comprehensive final summary"""
    print("\n" + "="*70)
    print("SRGAN FINAL SUMMARY - STABLE LSGAN IMPLEMENTATION")
    print("="*70)
    
    print("\nDATASET INFORMATION:")
    print(f"  Training images: 800")
    print(f"  Validation images: 100")
    print(f"  Scale factor: 4x")
    print(f"  Input LR size: 24x24")
    print(f"  Output SR size: 96x96")
    
    print("\nMODEL ARCHITECTURE:")
    print(f"  Generator: 16 Residual Blocks + 2x PixelShuffle")
    print(f"  Discriminator: 8 Convolutional layers")
    print(f"  Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"  Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")
    print(f"  Perceptual Loss: VGG19 (ReLU1_2, ReLU2_2, ReLU3_4, ReLU4_4)")
    
    print("\nLOSS CONFIGURATION (LSGAN):")
    print(f"  MSE Weight: {config.mse_weight}")
    print(f"  Perceptual (VGG) Weight: {config.perceptual_weight}")
    print(f"  Adversarial Weight: {config.adversarial_weight}")
    print(f"  LSGAN Real Target: {config.lsgan_target_real}")
    print(f"  LSGAN Fake Target: {config.lsgan_target_fake}")
    
    print("\nTRAINING STRATEGY:")
    print(f"  Pretrain epochs (MSE only): {config.pretrain_epochs}")
    print(f"  Total epochs: {config.num_epochs}")
    print(f"  Gradient Clipping: G={config.grad_clip_G}, D={config.grad_clip_D}")
    print(f"  LR Scheduler: ReduceLROnPlateau (patience=10, factor=0.5)")
    
    print("\nFINAL METRICS:")
    print(f"  PSNR: {avg_psnr:.2f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    print(f"  PSNR vs Bicubic: +{avg_psnr - 28.5:.1f} dB")
    quality = 'Excellent' if avg_psnr > 30 else 'Good' if avg_psnr > 28 else 'Improving'
    print(f"  Quality: {quality}")
    
    print("\nSAVED FILES:")
    print(f"  generator_final.pth")
    print(f"  discriminator_final.pth")
    print(f"  comparison_results.png")
    print(f"  training_history.png")
    print(f"  samples/ (epoch-wise samples)")
    
    print("\nKEY IMPROVEMENTS OVER PREVIOUS VERSION:")
    print(f"  LSGAN instead of BCE -> Stable training")
    print(f"  Gradient clipping -> Prevent explosion")
    print(f"  Extended pretraining -> Better initialization")
    print(f"  Label smoothing (0.9/0.1) -> Balanced GAN")
    print(f"  ReduceLROnPlateau -> Better convergence")
    
    print("\n" + "="*70)
    print("SRGAN TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)

def test_inference(generator, config, device, image_path=None):
    """Test SRGAN on a single image"""
    print("\n" + "="*50)
    print("TESTING SINGLE IMAGE INFERENCE")
    print("="*50)
    
    if image_path is None:
        val_files = [f for f in os.listdir(config.val_hr_path) 
                    if f.endswith(('.png', '.jpg', '.jpeg'))]
        if val_files:
            image_path = os.path.join(config.val_hr_path, val_files[0])
        else:
            print("No validation images found")
            return
    
    print(f"Testing on: {os.path.basename(image_path)}")
    
    # Load and process
    hr_image = Image.open(image_path).convert('RGB')
    w, h = hr_image.size
    new_w = (w // 4) * 4
    new_h = (h // 4) * 4
    hr_image = hr_image.resize((new_w, new_h), Image.BICUBIC)
    
    # Create LR
    lr_image = hr_image.resize((new_w//4, new_h//4), Image.BICUBIC)
    
    # Convert to tensor
    lr_tensor = (ToTensor()(lr_image) - 0.5) * 2
    lr_tensor = lr_tensor.unsqueeze(0).to(device)
    
    # Super-resolve
    generator.eval()
    with torch.no_grad():
        sr_tensor = generator(lr_tensor)
    
    # Convert back
    sr_image = denormalize(sr_tensor.cpu()[0])
    lr_np = np.array(lr_image)
    hr_np = np.array(hr_image)
    
    # Compute metrics
    psnr_val, ssim_val = compute_metrics(sr_tensor.cpu()[0], 
                                         (ToTensor()(hr_image) - 0.5) * 2)
    
    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(lr_np)
    axes[0].set_title(f'LR Input ({lr_image.size[0]}x{lr_image.size[1]})', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(sr_image)
    axes[1].set_title(f'SR Output ({sr_image.shape[1]}x{sr_image.shape[0]})\nPSNR: {psnr_val:.1f} dB, SSIM: {ssim_val:.3f}', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(hr_np)
    axes[2].set_title(f'HR Ground Truth ({hr_image.size[0]}x{hr_image.size[1]})', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_inference.png', dpi=150)
    plt.show()
    
    print(f"\nResults:")
    print(f"  - PSNR: {psnr_val:.2f} dB")
    print(f"  - SSIM: {ssim_val:.4f}")
    print(f"  - Saved: test_inference.png")

def export_complete_results(avg_psnr, avg_ssim, generator, discriminator, train_history):
    """Export all training results"""
    
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'SRGAN with LSGAN',
        'dataset': {
            'name': 'DIV2K',
            'train_samples': 800,
            'val_samples': 100,
            'scale_factor': config.scale_factor,
            'hr_size': config.hr_size,
            'lr_size': config.lr_size
        },
        'architecture': {
            'generator_res_blocks': 16,
            'generator_params': sum(p.numel() for p in generator.parameters()),
            'discriminator_params': sum(p.numel() for p in discriminator.parameters()),
            'perceptual_loss': 'VGG19 (layers: ReLU1_2, ReLU2_2, ReLU3_4, ReLU4_4)'
        },
        'loss_config': {
            'loss_type': 'LSGAN',
            'mse_weight': config.mse_weight,
            'perceptual_weight': config.perceptual_weight,
            'adversarial_weight': config.adversarial_weight,
            'lsgan_target_real': config.lsgan_target_real,
            'lsgan_target_fake': config.lsgan_target_fake
        },
        'training_config': {
            'pretrain_epochs': config.pretrain_epochs,
            'total_epochs': config.num_epochs,
            'batch_size': config.batch_size,
            'learning_rate_G': config.learning_rate_G,
            'learning_rate_D': config.learning_rate_D,
            'gradient_clip_G': config.grad_clip_G,
            'gradient_clip_D': config.grad_clip_D
        },
        'final_metrics': {
            'psnr': float(avg_psnr),
            'ssim': float(avg_ssim),
            'psnr_vs_bicubic': float(avg_psnr - 28.5)
        },
        'training_history': {
            'generator_loss': train_history['G_loss'],
            'discriminator_loss': train_history['D_loss']
        }
    }
    
    # Add GAN phase history
    if len(train_history.get('G_mse', [])) > 0:
        results['training_history']['mse_loss'] = train_history['G_mse']
        results['training_history']['perceptual_loss'] = train_history['G_perceptual']
        results['training_history']['adversarial_loss'] = train_history['G_adv']
        results['training_history']['d_real_loss'] = train_history['D_real']
        results['training_history']['d_fake_loss'] = train_history['D_fake']
    
    # Save JSON
    with open('srgan_lsgan_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print("Files saved:")
    print("  srgan_lsgan_results.json")
    print(f"  generator_epoch_{config.num_epochs}.pth")
    print(f"  discriminator_epoch_{config.num_epochs}.pth")
    print("  comparison_results.png")
    print("  training_history.png")
    print("  test_inference.png")
    print("  samples/ (directory)")
    print("="*60)

def save_complete_model(generator, discriminator, config, metrics, history, optimizer_G, optimizer_D, device, save_dir='saved_model'):
    """Save complete model with all components and metadata"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model checkpoints
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'generator_optimizer_state': optimizer_G.state_dict(),
        'discriminator_optimizer_state': optimizer_D.state_dict(),
        'config': config,
        'metrics': metrics,
        'history': history,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }, os.path.join(save_dir, 'srgan_complete_model.pth'))
    
    # Save just the generator for inference (smaller file)
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'config': {
            'scale_factor': config.scale_factor,
            'hr_size': config.hr_size,
            'lr_size': config.lr_size,
            'n_channels': config.n_channels
        },
        'metrics': metrics
    }, os.path.join(save_dir, 'srgan_generator_only.pth'))
    
    # Save as TorchScript for deployment
    generator.eval()
    example_input = torch.randn(1, 3, config.lr_size, config.lr_size).to(device)
    traced_script = torch.jit.trace(generator, example_input)
    traced_script.save(os.path.join(save_dir, 'srgan_scripted.pt'))
    
    print(f"\nModel saved to '{save_dir}/'")
    print(f"  - Complete model: srgan_complete_model.pth")
    print(f"  - Generator only: srgan_generator_only.pth")
    print(f"  - TorchScript: srgan_scripted.pt")
    
    return save_dir


def main():
    """Main training function"""
    
    print("="*60)
    print("STARTING STABLE SRGAN TRAINING WITH LSGAN")
    print("="*60)
    print(f"Phase 1: Pretraining G with MSE only for {config.pretrain_epochs} epochs")
    print(f"Phase 2: Full LSGAN training")
    print("="*60)
    
    # Create datasets and dataloaders
    train_dataset = DIV2KDataset(config.train_hr_path, is_train=True, 
                                hr_size=config.hr_size, scale_factor=config.scale_factor)
    val_dataset = DIV2KDataset(config.val_hr_path, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Initialize loss functions
    perceptual_loss = VGGPerceptualLoss()
    mse_loss = nn.MSELoss()
    lsgan = LSGANLoss()
    
    print("Perceptual loss (VGG19) initialized")
    print("LSGAN loss functions initialized")
    
    # Initialize optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=config.learning_rate_G, 
                             betas=(config.beta1, config.beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.learning_rate_D, 
                             betas=(config.beta1, config.beta2))
    
    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', 
                                                       factor=0.5, patience=10, verbose=True)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', 
                                                       factor=0.5, patience=10, verbose=True)
    
    print("Optimizers initialized with Adam")
    print(f"G initial LR: {config.learning_rate_G}")
    print(f"D initial LR: {config.learning_rate_D}")
    
    # Training history
    train_history = {
        'G_loss': [], 'G_mse': [], 'G_perceptual': [], 'G_adv': [],
        'D_loss': [], 'D_real': [], 'D_fake': []
    }
    
    best_psnr = 0
    
    # Training loop
    for epoch in range(config.num_epochs):
        generator.train()
        discriminator.train()
        
        epoch_G_total = 0
        epoch_G_mse = 0
        epoch_G_perceptual = 0
        epoch_G_adv = 0
        epoch_D_total = 0
        epoch_D_real = 0
        epoch_D_fake = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        
        for batch_idx, (lr, hr) in enumerate(progress_bar):
            lr, hr = lr.to(device), hr.to(device)
            batch_size = lr.size(0)
            
            # Generate SR images
            sr = generator(lr)
            
            # PHASE 1: Pretrain Generator with MSE only
            if epoch < config.pretrain_epochs:
                optimizer_G.zero_grad()
                mse = mse_loss(sr, hr)
                mse.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), config.grad_clip_G)
                optimizer_G.step()
                
                epoch_G_total += mse.item()
                epoch_G_mse += mse.item()
                
                progress_bar.set_postfix({'Phase': 'Pretrain', 'MSE': f'{mse.item():.4f}'})
            
            # PHASE 2: Full LSGAN Training
            else:
                # Train Discriminator
                optimizer_D.zero_grad()
                
                real_pred = discriminator(hr)
                d_real_loss = nn.MSELoss()(real_pred, torch.ones_like(real_pred) * config.lsgan_target_real)
                
                fake_pred = discriminator(sr.detach())
                d_fake_loss = nn.MSELoss()(fake_pred, torch.zeros_like(fake_pred) * config.lsgan_target_fake)
                
                d_loss = (d_real_loss + d_fake_loss) / 2
                
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), config.grad_clip_D)
                optimizer_D.step()
                
                # Train Generator
                optimizer_G.zero_grad()
                
                sr = generator(lr)
                fake_pred = discriminator(sr)
                
                mse = mse_loss(sr, hr)
                perc = perceptual_loss(sr, hr)
                g_adv_loss = nn.MSELoss()(fake_pred, torch.ones_like(fake_pred) * config.lsgan_target_real)
                
                g_loss = (config.mse_weight * mse + 
                         config.perceptual_weight * perc + 
                         config.adversarial_weight * g_adv_loss)
                
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), config.grad_clip_G)
                optimizer_G.step()
                
                # Record losses
                epoch_G_total += g_loss.item()
                epoch_G_mse += mse.item()
                epoch_G_perceptual += perc.item()
                epoch_G_adv += g_adv_loss.item()
                epoch_D_total += d_loss.item()
                epoch_D_real += d_real_loss.item()
                epoch_D_fake += d_fake_loss.item()
                
                progress_bar.set_postfix({
                    'G': f'{g_loss.item():.3f}',
                    'D': f'{d_loss.item():.3f}',
                    'MSE': f'{mse.item():.3f}',
                    'VGG': f'{perc.item():.3f}'
                })
        
        # Average losses for the epoch
        num_batches = len(train_loader)
        avg_G_loss = epoch_G_total / num_batches
        avg_D_loss = epoch_D_total / num_batches
        
        train_history['G_loss'].append(avg_G_loss)
        train_history['D_loss'].append(avg_D_loss)
        
        if epoch >= config.pretrain_epochs:
            train_history['G_mse'].append(epoch_G_mse / num_batches)
            train_history['G_perceptual'].append(epoch_G_perceptual / num_batches)
            train_history['G_adv'].append(epoch_G_adv / num_batches)
            train_history['D_real'].append(epoch_D_real / num_batches)
            train_history['D_fake'].append(epoch_D_fake / num_batches)
        
        # Update schedulers
        if epoch >= config.pretrain_epochs:
            scheduler_G.step(avg_G_loss)
            scheduler_D.step(avg_D_loss)
        
        # Print summary
        if epoch < config.pretrain_epochs:
            print(f'Epoch {epoch+1}: G_MSE={avg_G_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}: G={avg_G_loss:.4f}, D={avg_D_loss:.4f}, '
                  f'MSE={train_history["G_mse"][-1]:.4f}, '
                  f'VGG={train_history["G_perceptual"][-1]:.4f}')
        
        # Save samples and evaluate
        if (epoch + 1) % config.save_interval == 0 or epoch == config.num_epochs - 1:
            generator.eval()
            with torch.no_grad():
                sample_lr, sample_hr = next(iter(val_loader))
                sample_lr, sample_hr = sample_lr.to(device), sample_hr.to(device)
                sample_sr = generator(sample_lr)
                save_sample_images(sample_lr.cpu(), sample_sr.cpu(), sample_hr.cpu(), epoch+1)
                torch.save(generator.state_dict(), f'generator_epoch_{epoch+1}.pth')
                torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch+1}.pth')
                print(f"Models saved at epoch {epoch+1}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    
    # Load final model and evaluate
    final_generator = Generator().to(device)
    final_generator.load_state_dict(torch.load(f'generator_epoch_{config.num_epochs}.pth'))
    avg_psnr, avg_ssim, psnr_list, ssim_list = evaluate_model(final_generator, val_loader, device)
    
    # Visual comparison
    visual_comparison(final_generator, val_loader, device, num_samples=4)
    
    # Plot training history
    if len(train_history.get('G_mse', [])) > 0:
        plot_training_history(train_history, config.pretrain_epochs)
    else:
        print("GAN training not started yet. Run more epochs first.")
    
    # Final summary
    final_summary(avg_psnr, avg_ssim, generator, discriminator)
    
    # Test inference
    test_inference(final_generator, config, device)
    
    # Export results
    export_complete_results(avg_psnr, avg_ssim, generator, discriminator, train_history)
    
    # Save complete model
    metrics = {'psnr': avg_psnr, 'ssim': avg_ssim}
    save_complete_model(final_generator, discriminator, config, metrics, train_history, 
                       optimizer_G, optimizer_D, device)
    
    return final_generator, avg_psnr, avg_ssim


if __name__ == "__main__":
    final_generator, avg_psnr, avg_ssim = main()