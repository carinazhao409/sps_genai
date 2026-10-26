import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


import math

class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):
        """
        x: Tensor of shape (B, 1, 1, 1)
        returns: Tensor of shape (B, 1, 1, 2 * num_frequencies)
        """
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)         



def linear_diffusion_schedule(diffusion_times, min_rate=1e-4, max_rate=0.02, num_steps=1000):
    """
    diffusion_times: Tensor of any shape, with values in [0, 1)
    Returns:
        noise_rates, signal_rates: Tensors with the SAME shape as diffusion_times

    Note: alpha_bar is the cumulative product *along the diffusion schedule*. We therefore
    build the schedule once on a fixed grid of `num_steps` and then look up each sample's
    time on it. (Running `cumprod` directly on `diffusion_times` would multiply across the
    batch dimension, making each sample's noise level depend on the other samples.)
    """
    diffusion_times = diffusion_times.to(dtype=torch.float32)

    betas = torch.linspace(min_rate, max_rate, num_steps, device=diffusion_times.device)
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)

    idx = (diffusion_times * (num_steps - 1)).long().clamp(0, num_steps - 1)
    alpha_bars_t = alpha_bars[idx]

    signal_rates = torch.sqrt(alpha_bars_t)
    noise_rates = torch.sqrt(1.0 - alpha_bars_t)
    return noise_rates, signal_rates


def cosine_diffusion_schedule(diffusion_times):
    # diffusion_times: Tensor of shape [T] or [B] with values in [0, 1]
    signal_rates = torch.cos(diffusion_times * math.pi / 2)
    noise_rates = torch.sin(diffusion_times * math.pi / 2)
    return noise_rates, signal_rates


def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    # Flatten diffusion_times to handle any shape
    original_shape = diffusion_times.shape
    diffusion_times_flat = diffusion_times.flatten()

    # Compute start and end angles from signal rate bounds
    start_angle = torch.acos(torch.tensor(max_signal_rate, dtype=torch.float32, device=diffusion_times.device))
    end_angle = torch.acos(torch.tensor(min_signal_rate, dtype=torch.float32, device=diffusion_times.device))

    # Linearly interpolate angles
    diffusion_angles = start_angle + diffusion_times_flat * (end_angle - start_angle)

    # Compute signal and noise rates
    signal_rates = torch.cos(diffusion_angles).reshape(original_shape)
    noise_rates = torch.sin(diffusion_angles).reshape(original_shape)

    return noise_rates, signal_rates


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.needs_projection = in_channels != out_channels
        if self.needs_projection:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

        self.norm = nn.BatchNorm2d(in_channels, affine=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        residual = self.proj(x)
        # x = self.norm(x)
        x = self.swish(self.conv1(x))
        x = self.conv2(x)
        return x + residual

class DownBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        x = self.pool(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x



class UNet(nn.Module):
    def __init__(self, image_size, num_channels, embedding_dim=32):
        super().__init__()
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.num_channels = num_channels
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.embedding = SinusoidalEmbedding(num_frequencies=16)
        self.embedding_proj = nn.Conv2d(embedding_dim, 32, kernel_size=1)

        self.down1 = DownBlock(32, in_channels=64, block_depth=2)
        self.down2 = DownBlock(64, in_channels=32, block_depth=2)
        self.down3 = DownBlock(96, in_channels=64, block_depth=2) 

        self.mid1 = ResidualBlock(in_channels=96, out_channels=128)
        self.mid2 = ResidualBlock(in_channels=128, out_channels=128)

        self.up1 = UpBlock(96, in_channels=128, block_depth=2) 
        self.up2 = UpBlock(64, block_depth=2, in_channels=96)
        self.up3 = UpBlock(32, block_depth=2, in_channels=64)

        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)  # Keep zero init like TF reference

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        noise_emb = self.embedding(noise_variances)  # shape: (B, 1, 1, 32)
        # Upsample to match image size like TF reference
        noise_emb = F.interpolate(noise_emb.permute(0, 3, 1, 2), size=(self.image_size, self.image_size), mode='nearest')
        x = torch.cat([x, noise_emb], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips) 
        x = self.down3(x, skips)    

        x = self.mid1(x)     
        x = self.mid2(x)   

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final(x)



import copy

class DiffusionModel(nn.Module):
    def __init__(self, model, schedule_fn):
        super().__init__()
        self.network = model
        self.ema_network = copy.deepcopy(model)
        self.ema_network.eval()
        self.ema_decay = 0.8
        self.schedule_fn = schedule_fn
        self.normalizer_mean = 0.0
        self.normalizer_std = 1.0

    def to(self, device):
        # Override to() to ensure both networks move to the same device
        super().to(device)
        self.ema_network.to(device)
        return self

    def set_normalizer(self, mean, std):
        self.normalizer_mean = mean
        self.normalizer_std = std

    def denormalize(self, x):
        return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # Use EMA network for inference, main network for training
        if training:
            network = self.network
            network.train()
        else:
            network = self.ema_network
            network.eval()

        pred_noises = network(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        for step in range(diffusion_steps):
            t = torch.ones((initial_noise.shape[0], 1, 1, 1), device=initial_noise.device) * (1 - step * step_size)
            noise_rates, signal_rates = self.schedule_fn(t)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)

            next_diffusion_times = t - step_size
            next_noise_rates, next_signal_rates = self.schedule_fn(next_diffusion_times)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        return pred_images

    def generate(self, num_images, diffusion_steps, image_size=64, initial_noise=None):
        if initial_noise is None:
            initial_noise = torch.randn((num_images, self.network.num_channels, image_size, image_size), device=next(self.parameters()).device)
        with torch.no_grad():
            return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))

    def train_step(self, images, optimizer, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)

        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
        loss = loss_fn(pred_noises, noises)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # Update the EMA copy. This MUST run: `denoise(training=False)` - used by
        # generate(), test_step() and plot_images() - reads from `ema_network`, so if the
        # EMA is never updated, every sample and validation loss comes from an untrained net.
        with torch.no_grad():
            for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_param.copy_(self.ema_decay * ema_param + (1.0 - self.ema_decay) * param)
            # Buffers (e.g. BatchNorm running stats) are copied straight across.
            for ema_buf, buf in zip(self.ema_network.buffers(), self.network.buffers()):
                ema_buf.copy_(buf)

        return loss.item()

    def test_step(self, images, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)

        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        with torch.no_grad():
            pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
            loss = loss_fn(pred_noises, noises)

        return loss.item()

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            image_size=self.network.image_size,
            diffusion_steps=20,
        ).cpu()
        show_image(generated_images[0])      

