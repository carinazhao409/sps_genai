"""Train CIFAR-10 DDPM and EBM checkpoints used by the FastAPI endpoints."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from app.assignment4_models import DiffusionUNet, EnergyCNN, diffusion_loss, langevin_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['diffusion', 'energy', 'both'], default='both')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--diffusion-steps', type=int, default=200)
    parser.add_argument('--langevin-steps', type=int, default=20)
    parser.add_argument('--max-batches', type=int, default=0, help='Optional smoke-test limit; 0 means full dataset')
    parser.add_argument('--data-dir', default='data')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.diffusion_steps < 2 or args.langevin_steps < 1:
        parser.error('epochs/batch-size/langevin-steps must be positive; diffusion-steps must be >= 2')
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    for kind in (['diffusion', 'energy'] if args.model == 'both' else [args.model]):
        model = (DiffusionUNet() if kind == 'diffusion' else EnergyCNN()).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for index, (real, _) in enumerate(loader):
                if args.max_batches and index >= args.max_batches:
                    break
                real = real.to(device)
                optimizer.zero_grad(set_to_none=True)
                if kind == 'diffusion':
                    loss = diffusion_loss(model, real, args.diffusion_steps)
                else:
                    # Contrastive divergence: decrease real-image energy, increase sampled-image energy.
                    negative = langevin_sample(model, count=real.size(0), steps=args.langevin_steps, device=device)
                    loss = model(real).mean() - model(negative).mean()
                    # Small energy regularizer helps keep the scale under control.
                    loss = loss + 0.001 * (model(real).square().mean() + model(negative).square().mean())
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if not losses:
                raise RuntimeError('No training batches processed')
            print(f'{kind} epoch {epoch+1}/{args.epochs}: mean loss={sum(losses)/len(losses):.4f}', flush=True)
        target = Path('app') / f'cifar10_{kind}.pth'
        torch.save({'state_dict': model.cpu().state_dict(), 'model': kind,
                    'diffusion_steps': args.diffusion_steps if kind == 'diffusion' else None,
                    'epochs': args.epochs, 'max_batches': args.max_batches}, target)
        print(f'Saved {target}', flush=True)


if __name__ == '__main__':
    main()
