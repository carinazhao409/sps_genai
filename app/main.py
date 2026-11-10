"""Combined FastAPI application for the earlier modules and Assignment 2."""

import io
from pathlib import Path

import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.cnn_model import CNN
from app.embedding_model import EmbeddingModel
from app.gan_model import Generator
from app.rnn_text import generate_text as generate_lstm_text, load_model as load_lstm_model


app = FastAPI(
    title="Generative AI and CNN Image Classifier API",
    description="Bigram text generation, word embeddings, and CIFAR-10 classification",
    version="2.0",
)

# Original bigram model and sample corpus
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)
embedding_model = None


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    global embedding_model
    if embedding_model is None:
        try:
            embedding_model = EmbeddingModel()
        except OSError as exc:
            raise HTTPException(status_code=503, detail='spaCy en_core_web_lg model is not installed.') from exc
    embedding = embedding_model.calculate_embedding(request.word)
    return {"word": request.word, "embedding": embedding}


# Assignment 2: CNN image classifier
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cifar10_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

cnn_model = CNN().to(DEVICE)
if MODEL_PATH.exists():
    cnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    cnn_model.eval()
else:
    cnn_model = None

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Classify an uploaded image into one of the ten CIFAR-10 classes."""
    try:
        image_data = await file.read()
        with Image.open(io.BytesIO(image_data)) as uploaded_image:
            image = uploaded_image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Please upload a valid image.") from exc
    finally:
        await file.close()

    if cnn_model is None:
        raise HTTPException(status_code=503, detail='CNN checkpoint missing; train the earlier CNN first.')
    image_tensor = image_transform(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        outputs = cnn_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    return {
        "filename": file.filename,
        "predicted_class": CLASSES[predicted.item()],
        "confidence": round(confidence.item(), 4),
    }


# Assignment 3: generate handwritten digit images with the MNIST GAN.
# Train first: uv run python train_gan.py --epochs 10
from fastapi.responses import Response

GAN_PATH = BASE_DIR / 'mnist_gan_generator.pth'
_gan_generator = None


def load_gan_generator():
    global _gan_generator
    if _gan_generator is None:
        if not GAN_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail='GAN weights not found. Run: uv run python train_gan.py --epochs 10',
            )
        model = Generator(z_dim=100).to(DEVICE)
        model.load_state_dict(torch.load(GAN_PATH, map_location=DEVICE, weights_only=True))
        model.eval()
        _gan_generator = model
    return _gan_generator


@app.get('/generate-digit', response_class=Response)
def generate_digit():
    """Return one GAN-generated 28x28 grayscale PNG image."""
    generator = load_gan_generator()
    with torch.inference_mode():
        noise = torch.randn(1, 100, device=DEVICE)
        image_tensor = generator(noise).squeeze().cpu()
        image_uint8 = ((image_tensor + 1) / 2 * 255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_uint8, mode='L')
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return Response(content=buffer.getvalue(), media_type='image/png')


# Module 7: character-level LSTM text generation.
# Load the trained model once rather than for every request.
LSTM_PATH = BASE_DIR / "lstm_checkpoint.pt"
_lstm_model = _lstm_stoi = _lstm_itos = None


@app.post("/generate_with_rnn")
def generate_with_rnn(request: TextGenerationRequest):
    """Generate additional characters from the supplied seed using the trained LSTM."""
    if not request.start_word or not 1 <= request.length <= 1000:
        raise HTTPException(status_code=422, detail="Provide a nonempty seed and length between 1 and 1000")
    global _lstm_model, _lstm_stoi, _lstm_itos
    if _lstm_model is None:
        if not LSTM_PATH.exists():
            raise HTTPException(status_code=503, detail='LSTM checkpoint missing; train the earlier LSTM first.')
        _lstm_model, _lstm_stoi, _lstm_itos = load_lstm_model(LSTM_PATH)
    try:
        result = generate_lstm_text(
            _lstm_model, _lstm_stoi, _lstm_itos, request.start_word, request.length
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"generated_text": result}


# Module 9: fine-tuned GPT-2 question answering.


@app.post("/generate_with_llm")
def generate_with_llm(request: TextGenerationRequest):
    """Answer a question using locally fine-tuned GPT-2 weights."""
    if not request.start_word.strip() or not 1 <= request.length <= 256:
        raise HTTPException(status_code=422, detail="Provide a question and length between 1 and 256")
    try:
        from app.gpt2_qa import generate_answer
        result = generate_answer(request.start_word, request.length)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"generated_text": result}


# Assignment 4: CIFAR-10 image generators.
from app.assignment4_models import DiffusionUNet, EnergyCNN, sample_diffusion, langevin_sample

_image_generators = {}


def load_image_generator(kind):
    if kind not in _image_generators:
        path = BASE_DIR / f'cifar10_{kind}.pth'
        if not path.exists():
            raise HTTPException(status_code=503, detail=f'{kind} checkpoint missing. Run: python train_assignment4.py --model {kind}')
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
        model = (DiffusionUNet() if kind == 'diffusion' else EnergyCNN()).to(DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        _image_generators[kind] = (model, checkpoint)
    return _image_generators[kind]


def png_response(tensor):
    pixels = ((tensor[0].detach().cpu().permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().numpy()
    image = Image.fromarray(pixels, mode='RGB')
    output = io.BytesIO()
    image.save(output, format='PNG')
    return Response(content=output.getvalue(), media_type='image/png')


@app.get('/generate-diffusion', response_class=Response)
def generate_diffusion():
    """Generate a CIFAR-10-sized RGB image from a trained diffusion model."""
    model, checkpoint = load_image_generator('diffusion')
    image = sample_diffusion(model, steps=checkpoint['diffusion_steps'], device=DEVICE)
    return png_response(image)


@app.get('/generate-energy', response_class=Response)
def generate_energy():
    """Generate a CIFAR-10-sized RGB image by minimizing learned energy."""
    model, _ = load_image_generator('energy')
    image = langevin_sample(model, device=DEVICE)
    return png_response(image)
