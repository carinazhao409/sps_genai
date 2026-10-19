import torch
from app.gan_model import Generator, Discriminator, GAN


def test_generator_shape_and_range():
    model = Generator()
    with torch.no_grad():
        output = model(torch.randn(4, 100))
    assert output.shape == (4, 1, 28, 28)
    assert output.min() >= -1 and output.max() <= 1


def test_discriminator_shape_and_range():
    model = Discriminator()
    with torch.no_grad():
        output = model(torch.randn(4, 1, 28, 28))
    assert output.shape == (4, 1)
    assert output.min() >= 0 and output.max() <= 1


def test_gan_forward():
    with torch.no_grad():
        assert GAN()(torch.randn(2, 100)).shape == (2, 1, 28, 28)
