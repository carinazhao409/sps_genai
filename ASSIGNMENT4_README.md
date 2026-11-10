# Assignment 4 — CIFAR-10 Diffusion and Energy-Based Models

This update **extends the uploaded FastAPI project**, retaining its earlier routes. It adds a small DDPM-style noise-prediction U-Net and a convolutional energy-based model (EBM), both trained on **CIFAR-10 RGB images (32 × 32)**. The models are intentionally compact so the code is understandable; image quality depends heavily on training time and compute.

## 1. Install and train

From the `sps_genai` directory:

```bash
uv sync --python 3.13
uv run python train_assignment4.py --model both --epochs 10 --batch-size 64
```

Training downloads CIFAR-10 into `data/` (internet needed on the training machine). The script automatically uses CUDA if available, otherwise CPU. Training both models on CPU can be slow. The default diffusion schedule uses 200 timesteps. For a quick **code smoke test only**, run `uv run python train_assignment4.py --model both --epochs 1 --max-batches 2 --diffusion-steps 10 --langevin-steps 2`; **do not treat this as adequately trained**. Increase epochs and evaluate visual quality for the actual submission.

The training script saves two checkpoint files:

- `app/cifar10_diffusion.pth`
- `app/cifar10_energy.pth`

**Important:** the original `.gitignore` excludes `*.pth` and `*.pt`. Training artifacts must be copied onto the deployment machine or deliberately tracked with Git LFS (if allowed). Simply committing source code does not make trained endpoints work on the instructor's machine. The uploaded project also lacks its earlier CNN/LSTM/GAN/GPT-2 checkpoints; their endpoints return 503 until their weights are supplied.

## 2. Run the API locally

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` or request:

```bash
curl -o diffusion.png http://localhost:8000/generate-diffusion
curl -o energy.png http://localhost:8000/generate-energy
```

Both endpoints return `image/png` with a **32 × 32 RGB image**. If a checkpoint is missing, they return HTTP **503** with an actionable explanation instead of silently returning an untrained result.

## 3. Docker

Train locally first. The Dockerfile copies the `app/` directory, **including trained checkpoint files that are physically present there at build time**.

```bash
docker build -t sps-genai-assignment4 .
docker run --rm -p 8000:80 sps-genai-assignment4
curl -o diffusion.png http://localhost:8000/generate-diffusion
curl -o energy.png http://localhost:8000/generate-energy
```

The Docker build downloads Python packages and the spaCy language model; internet access is needed. If Docker is unavailable on your machine, these commands still need to be run and verified before claiming deployment success.

## 4. Tests

```bash
uv run --with pytest pytest -q tests/test_assignment4.py
```

Tests check output dimensions, backpropagation, Langevin sampling, HTTP status, PNG format, and endpoint behavior with **temporary randomly initialized checkpoints**. These tests confirm plumbing, **not training quality**.

## 5. GitHub submission

```bash
git status
git add app/main.py app/assignment4_models.py train_assignment4.py tests/test_assignment4.py ASSIGNMENT4_README.md Dockerfile
# If this is the first commit of the entire project, stage the other required source files too.
git commit -m "Add CIFAR-10 diffusion and energy models with FastAPI endpoints"
git push
```

Confirm the remote repository and branch before pushing. Check that the instructor can obtain **both trained weights** and that a clean Docker build can serve both endpoints. Do not commit downloaded CIFAR-10 data. Do not claim a GitHub push has occurred until you execute it successfully.

## How the algorithms work

**Diffusion:** Sample a timestep, add Gaussian noise according to a fixed variance schedule, and train the U-Net to predict the added noise using mean-squared error. At inference, begin with Gaussian noise and iteratively denoise.

**Energy model:** Train a scalar-valued CNN to assign lower energies to real images and higher energies to model-generated negatives. Negatives are obtained using Langevin updates that compute gradients **with respect to image pixels**, not network parameters. At inference, Langevin updates generate new images from random initialization.

## Limitations

These are teaching implementations rather than state-of-the-art generators. EBM sampling is sensitive to step size, number of updates, and noise; DDPM sampling may be slow on CPU. There is no claim of pretrained model accuracy or visual quality until full training is run and images are inspected.
