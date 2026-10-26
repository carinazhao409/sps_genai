import torch
import matplotlib.pyplot as plt
import math
import numpy as np

def generate_samples(
    model,
    device,
    num_samples=10,
    latent_dim=2
):
    model = model.to(device)
    model.eval()

    if num_samples < 1:
        raise ValueError(
            "num_samples must be at least 1"
        )

    # Identify whether this is a GAN
    is_gan = (
        hasattr(model, "generator")
        and hasattr(model, "critic")
    )

    if is_gan:
        latent_dim = model.z_dim

    with torch.no_grad():
        z = torch.randn(
            num_samples,
            latent_dim,
            device=device
        )

        if is_gan:
            samples = model.generator(z)

            # Convert [-1, 1] to [0, 1]
            samples = (samples + 1) / 2
        else:
            samples = model.decoder(z)

        samples = samples.cpu().numpy()

    cols = min(5, num_samples)
    rows = math.ceil(num_samples / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 2, rows * 2)
    )

    axes = np.atleast_1d(axes).flatten()

    for i in range(num_samples):
        image = samples[i].transpose(1, 2, 0)

        if image.shape[-1] == 1:
            axes[i].imshow(
                image[:, :, 0],
                cmap="gray",
                vmin=0,
                vmax=1
            )
        else:
            axes[i].imshow(
                image.clip(0, 1)
            )

        axes[i].axis("off")

    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def generate_diffusion_samples(model, device, num_samples=10, diffusion_steps=100):
    """Sample from the diffusion model and display an image grid."""
    if num_samples < 1 or diffusion_steps < 1:
        raise ValueError("num_samples and diffusion_steps must be positive")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        samples = model.generate(num_images=num_samples, diffusion_steps=diffusion_steps,
                                 image_size=model.network.image_size).cpu()
    cols = min(5, num_samples)
    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.atleast_1d(axes).flatten()
    for i, sample in enumerate(samples):
        image = sample.permute(1, 2, 0).numpy()
        if image.shape[-1] == 1:
            axes[i].imshow(image[:, :, 0], cmap="gray", vmin=0, vmax=1)
        else:
            axes[i].imshow(image.clip(0, 1))
        axes[i].axis("off")
    for ax in axes[num_samples:]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    return samples
