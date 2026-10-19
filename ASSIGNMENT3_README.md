# Assignment 3: GAN on MNIST

## Included
- `app/gan_model.py`: exactly specified generator/discriminator architecture (Part 1)
- `train_gan.py`: standard GAN adversarial training on MNIST (Part 2)
- `app/main.py`: original API with `GET /generate-digit` added
- `tests/test_gan.py`: shape and output-range tests

## Setup (Windows terminal, repository root)
```powershell
uv sync
uv run python -m pytest tests/test_gan.py
uv run python train_gan.py --epochs 10
uv run uvicorn app.main:app --reload
```
If pytest is not installed, run `uv run --with pytest python -m pytest tests/test_gan.py` instead.

Visit `http://127.0.0.1:8000/docs` and use `GET /generate-digit`. The endpoint returns a PNG, not a digit class label. You can also open `http://127.0.0.1:8000/generate-digit` in a browser.

**Important:** The trained `app/mnist_gan_generator.pth` file is NOT included; it must be produced by running training. The included `app/cifar10_cnn.pth` is from the previous assignment. MNIST data downloads during training. No model-quality claim is made until training and sample inspection are completed.

## Why this differs from the class helper WGAN
The helper's WGAN uses a critic, RMSprop, weight clipping and Wasserstein losses. Assignment 3 instead requires a discriminator outputting a probability. Here we use Sigmoid and binary cross-entropy, with alternating discriminator/generator updates. The generator's Tanh output is matched with MNIST normalized to [-1,1].

## GitHub (after testing and training)
```powershell
git add app/gan_model.py app/main.py train_gan.py tests/test_gan.py ASSIGNMENT3_README.md
git add app/mnist_gan_generator.pth
git commit -m "Implement and deploy MNIST GAN for Assignment 3"
git push
```
If `.gitignore` ignores `.pth`, use `git add -f app/mnist_gan_generator.pth`. Do not commit the downloaded MNIST data, local virtual environment, or all sample images.

## Docker
The existing Dockerfile should be checked before deployment; make sure it copies `app/mnist_gan_generator.pth` and starts `uvicorn app.main:app`. The endpoint returns 503 until trained weights are present.
