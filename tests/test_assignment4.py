"""Assignment 4 model and API smoke tests (do not require CIFAR downloads)."""
import io
import torch
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app
from app.assignment4_models import DiffusionUNet, EnergyCNN, diffusion_loss, langevin_sample


def test_model_shapes_and_gradients():
    torch.manual_seed(42)
    images = torch.randn(2, 3, 32, 32)
    diffusion = DiffusionUNet()
    loss = diffusion_loss(diffusion, images, steps=8)
    loss.backward()
    assert torch.isfinite(loss)
    assert diffusion.input.weight.grad is not None
    energy = EnergyCNN()
    assert energy(images).shape == (2,)
    samples = langevin_sample(energy, count=1, steps=2)
    assert samples.shape == (1, 3, 32, 32)
    assert samples.min() >= -1 and samples.max() <= 1


def test_endpoints_with_temporary_checkpoints(tmp_path, monkeypatch):
    import app.main as main
    monkeypatch.setattr(main, 'BASE_DIR', tmp_path)
    main._image_generators.clear()
    for kind, model in [('diffusion', DiffusionUNet()), ('energy', EnergyCNN())]:
        torch.save({'state_dict': model.state_dict(), 'diffusion_steps': 2}, tmp_path / f'cifar10_{kind}.pth')
    client = TestClient(app)
    for endpoint in ('/generate-diffusion', '/generate-energy'):
        response = client.get(endpoint)
        assert response.status_code == 200
        assert response.headers['content-type'] == 'image/png'
        with Image.open(io.BytesIO(response.content)) as image:
            assert image.size == (32, 32)
            assert image.mode == 'RGB'
    main._image_generators.clear()
