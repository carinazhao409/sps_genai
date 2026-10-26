import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_model(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10
):
    # Move model to selected device
    model = model.to(device)

    # Repeat the training process for each epoch
    for epoch in range(epochs):

        # Set model to training mode
        model.train()

        running_loss = 0.0

        for images, labels in data_loader:

            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Clear old gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        # Calculate average loss for this epoch
        avg_loss = running_loss / len(data_loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {avg_loss:.4f}"
        )

    return model

def vae_loss_function(recon_x, x, mu, logvar):

    beta = 500

    BCE = F.binary_cross_entropy(
        recon_x,
        x,
        reduction='sum'
    )

    KLD = -0.5 * torch.sum(
        1 + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    return beta * BCE + KLD


def train_vae_model(
    model,
    data_loader,
    criterion,
    optimizer,
    device='cpu',
    epochs=10
):

    model.to(device)
    model.train()

    for epoch in range(epochs):

        running_loss = 0.0

        train_loader_with_progress = tqdm(
            data_loader,
            desc=f'Epoch {epoch + 1}/{epochs}'
        )

        for data in train_loader_with_progress:

            inputs = data[0].to(device)

            optimizer.zero_grad()

            recon, mu, logvar = model(inputs)

            loss = criterion(
                recon,
                inputs,
                mu,
                logvar
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            train_loader_with_progress.set_postfix(
                loss=f'{loss.item():.4f}'
            )

    print("Finished Training")

    return model

def train_gan(
    model,
    data_loader,
    criterion=None,
    optimizer=None,
    device='cpu',
    epochs=10,
    lr=5e-5,
    n_critic=5,
    clip_value=0.01
):
    model = model.to(device)

    # Create separate optimizers
    if optimizer is None:
        optimizer_g = torch.optim.RMSprop(
            model.generator.parameters(),
            lr=lr
        )
        optimizer_c = torch.optim.RMSprop(
            model.critic.parameters(),
            lr=lr
        )
    elif isinstance(optimizer, dict):
        optimizer_g = optimizer["generator"]
        optimizer_c = optimizer["critic"]
    else:
        optimizer_g, optimizer_c = optimizer

    if n_critic < 1 or clip_value <= 0:
        raise ValueError(
            "n_critic and clip_value must be positive"
        )

    for epoch in range(epochs):
        model.train()

        total_c_loss = 0.0
        total_g_loss = 0.0
        batch_count = 0

        for batch in tqdm(data_loader):
            if isinstance(batch, (tuple, list)):
                real_images = batch[0]
            else:
                real_images = batch

            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Train the critic multiple times
            for _ in range(n_critic):
                noise = torch.randn(
                    batch_size,
                    model.z_dim,
                    device=device
                )

                fake_images = model.generator(
                    noise
                ).detach()

                real_score = model.critic(
                    real_images
                ).mean()

                fake_score = model.critic(
                    fake_images
                ).mean()

                critic_loss = -(
                    real_score - fake_score
                )

                optimizer_c.zero_grad()
                critic_loss.backward()
                optimizer_c.step()

                # WGAN weight clipping
                with torch.no_grad():
                    for param in model.critic.parameters():
                        param.clamp_(
                            -clip_value,
                            clip_value
                        )

            # Train the generator
            noise = torch.randn(
                batch_size,
                model.z_dim,
                device=device
            )

            fake_images = model.generator(noise)

            generator_loss = -model.critic(
                fake_images
            ).mean()

            optimizer_g.zero_grad()
            generator_loss.backward()
            optimizer_g.step()

            total_c_loss += critic_loss.item()
            total_g_loss += generator_loss.item()
            batch_count += 1

        if batch_count == 0:
            raise ValueError("data_loader is empty")

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Critic Loss: {total_c_loss / batch_count:.4f} | "
            f"Generator Loss: {total_g_loss / batch_count:.4f}"
        )

    print("Finished GAN Training")
    return model

def train_diffusion(model, data_loader, criterion, optimizer, device="cpu", epochs=10):
    """Train a diffusion noise predictor using the notebook's train_step method."""
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in data_loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device)
            loss = model.train_step(images, optimizer, criterion)
            total_loss += loss
            batches += 1
        if not batches:
            raise ValueError("data_loader must contain at least one batch")
        print(f"Diffusion Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / batches:.4f}")
    return model
