import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            32 * 32 * 3,
            200
        )

        self.fc2 = nn.Linear(
            200,
            150
        )

        self.fc3 = nn.Linear(
            150,
            10
        )

    def forward(self, x):
        x = self.flatten(x)

        x = torch.relu(
            self.fc1(x)
        )

        x = torch.relu(
            self.fc2(x)
        )

        x = torch.softmax(
            self.fc3(x),
            dim=1
        )

        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            padding=1
        )

        self.fc1 = nn.Linear(
            32 * 8 * 8,
            128
        )

        self.fc2 = nn.Linear(
            128,
            10
        )

    def forward(self, x):
        x = self.pool(
            F.relu(self.conv1(x))
        )

        x = self.pool(
            F.relu(self.conv2(x))
        )

        x = x.view(
            -1,
            32 * 8 * 8
        )

        x = F.relu(
            self.fc1(x)
        )

        x = self.fc2(x)

        return x


class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()

        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(32)

        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(64)

        # Convolutional Layer 4
        self.conv4 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            padding=1
        )

        self.bn4 = nn.BatchNorm2d(128)

        # Fully connected layers
        self.fc1 = nn.Linear(
            128 * 2 * 2,
            128
        )

        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(
            128,
            10
        )

    def forward(self, x):
        x = self.pool(
            F.relu(self.bn1(self.conv1(x)))
        )

        x = self.pool(
            F.relu(self.bn2(self.conv2(x)))
        )

        x = self.pool(
            F.relu(self.bn3(self.conv3(x)))
        )

        x = self.pool(
            F.relu(self.bn4(self.conv4(x)))
        )

        x = x.view(
            -1,
            128 * 2 * 2
        )

        x = F.relu(
            self.fc1(x)
        )

        x = self.dropout(x)

        x = self.fc2(x)

        return x



class VariationalEncoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(VariationalEncoder, self).__init__()

        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1
        )

        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=2, padding=1
        )

        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=3, stride=2, padding=1
        )

        self.flatten = nn.Flatten()

        self.fc_mu = nn.Linear(
            128 * 4 * 4, latent_dim
        )

        self.fc_logvar = nn.Linear(
            128 * 4 * 4, latent_dim
        )

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.flatten(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(
            latent_dim, 128 * 4 * 4
        )

        self.convtrans1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

        self.convtrans2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

        self.convtrans3 = nn.ConvTranspose2d(
            32,
            1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

    def forward(self, x):

        x = self.fc(x)

        x = x.view(-1, 128, 4, 4)

        x = F.relu(self.convtrans1(x))
        x = F.relu(self.convtrans2(x))

        x = torch.sigmoid(
            self.convtrans3(x)
        )

        return x

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x):

        mu, logvar = self.encoder(x)

        z = self.reparameterize(
            mu,
            logvar
        )

        reconstruction = self.decoder(z)

        return reconstruction, mu, logvar

class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Flatten()
        )

    def forward(self, x):
        return self.layers(x)

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()

        self.z_dim = z_dim

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                z_dim, 512, 4, 1, 0, bias=False
            ),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                512, 256, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                256, 128, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                128, 64, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                64, 3, 4, 2, 1
            ),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.reshape(
            z.shape[0],
            self.z_dim,
            1,
            1
        )
        return self.layers(z)

class GAN(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()

        self.z_dim = z_dim
        self.generator = Generator(z_dim)
        self.critic = Critic()

    def forward(self, z):
        return self.generator(z)

def get_model(model_name):

    if model_name == "FCNN":
        model = FCNN()

    elif model_name == "CNN":
        model = SimpleCNN()

    elif model_name == "EnhancedCNN":
        model = EnhancedCNN()

    elif model_name == "VAE":

        latent_dim = 2

        encoder = VariationalEncoder(
            latent_dim=latent_dim
        )

        decoder = Decoder(
            latent_dim=latent_dim
        )

        model = VAE(
            encoder,
            decoder
        )

    elif model_name == "Diffusion":
        from .diffusion import UNet, DiffusionModel, offset_cosine_diffusion_schedule
        model = DiffusionModel(
            UNet(image_size=64, num_channels=3, embedding_dim=32),
            offset_cosine_diffusion_schedule,
        )

    elif model_name == "GAN":
        model = GAN(z_dim=100)

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model