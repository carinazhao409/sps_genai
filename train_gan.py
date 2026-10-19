"""Train the assignment's GAN on MNIST and save weights for the API.
Run from repository root: uv run python train_gan.py --epochs 10
"""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from app.gan_model import GAN


def train(epochs=10, batch_size=128, lr=0.0002, z_dim=100, data_dir='data', seed=42):
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training on:', device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = GAN(z_dim=z_dim).to(device)
    generator, discriminator = model.generator, model.discriminator
    loss_fn = nn.BCELoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    samples_dir = Path('gan_samples')
    samples_dir.mkdir(exist_ok=True)
    fixed_noise = torch.randn(25, z_dim, device=device)

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        d_total = g_total = 0.0
        for real_images, _ in loader:
            real_images = real_images.to(device)
            b = real_images.size(0)
            real_labels = torch.ones(b, 1, device=device)
            fake_labels = torch.zeros(b, 1, device=device)

            # Update discriminator using real and detached generated images.
            opt_d.zero_grad()
            fake_images = generator(torch.randn(b, z_dim, device=device))
            d_loss = (loss_fn(discriminator(real_images), real_labels)
                      + loss_fn(discriminator(fake_images.detach()), fake_labels)) / 2
            d_loss.backward()
            opt_d.step()

            # Update generator; keep discriminator weights fixed for this step.
            opt_g.zero_grad()
            for param in discriminator.parameters():
                param.requires_grad_(False)
            generated = generator(torch.randn(b, z_dim, device=device))
            g_loss = loss_fn(discriminator(generated), real_labels)
            g_loss.backward()
            opt_g.step()
            for param in discriminator.parameters():
                param.requires_grad_(True)

            d_total += d_loss.item()
            g_total += g_loss.item()

        print(f'Epoch {epoch}/{epochs} | D loss: {d_total/len(loader):.4f} | G loss: {g_total/len(loader):.4f}')
        generator.eval()
        with torch.no_grad():
            sample = generator(fixed_noise).cpu()
        save_image((sample + 1) / 2, samples_dir / f'epoch_{epoch:02d}.png', nrow=5)

    output = Path('app/mnist_gan_generator.pth')
    torch.save(generator.cpu().state_dict(), output)
    print('Saved trained generator to', output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0002)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
