"""Compact CIFAR-10 diffusion and energy models for Assignment 4."""
import math
import torch
from torch import nn
import torch.nn.functional as F


def time_embedding(t, dim):
    half = dim // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    phases = t.float()[:, None] * frequencies[None, :]
    embedding = torch.cat([phases.sin(), phases.cos()], dim=1)
    return F.pad(embedding, (0, dim - embedding.shape[1]))


class ResidualBlock(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(8, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1))
        self.time_projection = nn.Linear(time_dim, channels)

    def forward(self, x, embedding):
        return F.silu(x + self.net(x + self.time_projection(embedding)[:, :, None, None]))


class DiffusionUNet(nn.Module):
    def __init__(self, channels=32, time_dim=64):
        super().__init__()
        self.input = nn.Conv2d(3, channels, 3, padding=1)
        self.down1 = ResidualBlock(channels, time_dim)
        self.down2 = ResidualBlock(channels * 2, time_dim)
        self.down3 = ResidualBlock(channels * 4, time_dim)
        self.ds1 = nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1)
        self.ds2 = nn.Conv2d(channels * 2, channels * 4, 4, stride=2, padding=1)
        self.us1 = nn.ConvTranspose2d(channels * 4, channels * 2, 4, stride=2, padding=1)
        self.us2 = nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1)
        self.up1 = ResidualBlock(channels * 2, time_dim)
        self.up2 = ResidualBlock(channels, time_dim)
        self.merge1 = nn.Conv2d(channels * 4, channels * 2, 1)
        self.merge2 = nn.Conv2d(channels * 2, channels, 1)
        self.output = nn.Conv2d(channels, 3, 3, padding=1)
        self.time_dim = time_dim

    def forward(self, x, t):
        emb = time_embedding(t, self.time_dim)
        a = self.down1(self.input(x), emb)
        b = self.down2(self.ds1(a), emb)
        c = self.down3(self.ds2(b), emb)
        d = self.up1(self.merge1(torch.cat([self.us1(c), b], dim=1)), emb)
        e = self.up2(self.merge2(torch.cat([self.us2(d), a], dim=1)), emb)
        return self.output(e)


class EnergyCNN(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(channels * 2, channels * 4, 4, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels * 4, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def diffusion_schedule(steps, device):
    betas = torch.linspace(1e-4, 0.02, steps, device=device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def diffusion_loss(model, clean, steps=200):
    _, _, alpha_bars = diffusion_schedule(steps, clean.device)
    t = torch.randint(0, steps, (clean.size(0),), device=clean.device)
    noise = torch.randn_like(clean)
    alpha_bar = alpha_bars[t, None, None, None]
    noisy = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise
    return F.mse_loss(model(noisy, t), noise)


@torch.no_grad()
def sample_diffusion(model, count=1, steps=200, device='cpu'):
    betas, alphas, alpha_bars = diffusion_schedule(steps, device)
    x = torch.randn(count, 3, 32, 32, device=device)
    model.eval()
    for step in reversed(range(steps)):
        t = torch.full((count,), step, device=device, dtype=torch.long)
        noise_prediction = model(x, t)
        x = (x - betas[step] / (1 - alpha_bars[step]).sqrt() * noise_prediction) / alphas[step].sqrt()
        if step > 0:
            x = x + betas[step].sqrt() * torch.randn_like(x)
    return x.clamp(-1, 1)


def langevin_sample(model, count=1, steps=60, step_size=0.1, noise_scale=0.01, device='cpu'):
    """Update image inputs, not model weights, using negative energy gradients."""
    model.eval()
    x = torch.empty(count, 3, 32, 32, device=device).uniform_(-1, 1)
    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        energy = model(x).sum()
        gradient = torch.autograd.grad(energy, x)[0]
        with torch.no_grad():
            x = (x - step_size * gradient + noise_scale * torch.randn_like(x)).clamp(-1, 1)
    return x.detach()
