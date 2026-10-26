"""Character-level LSTM adapted from the Module 6 RNN practical."""
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

DEFAULT_TEXT = 'this is a simple test dataset to train a small lstm model for text generation.'

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        return self.fc(x), hidden

class TextDataset(Dataset):
    def __init__(self, data, seq_len=30):
        self.data, self.seq_len = data, seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        n = self.seq_len
        return (torch.tensor(self.data[idx:idx+n], dtype=torch.long),
                torch.tensor(self.data[idx+1:idx+n+1], dtype=torch.long))

def train_and_save(path='lstm_checkpoint.pt', text=DEFAULT_TEXT, epochs=15):
    torch.manual_seed(42)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    encoded = [stoi[c] for c in text]
    dataset = TextDataset(encoded)
    if not len(dataset):
        raise ValueError('Training text must contain more than 30 characters')
    loader = DataLoader(dataset, batch_size=6, shuffle=True)
    model = LSTMModel(len(chars))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, len(chars)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f'Epoch {epoch+1}/{epochs}: average loss {total/len(loader):.4f}', flush=True)
    torch.save({'state_dict': model.state_dict(), 'chars': chars}, path)
    return path

def load_model(path='lstm_checkpoint.pt'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    chars = checkpoint['chars']
    model = LSTMModel(len(chars))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, {c: i for i, c in enumerate(chars)}, {i: c for i, c in enumerate(chars)}

def generate_text(model, stoi, itos, seed_text, length=100, temperature=1.0):
    if not seed_text:
        raise ValueError('start_word cannot be empty')
    unknown = sorted(set(seed_text) - set(stoi))
    if unknown:
        raise ValueError(f'Characters not in training vocabulary: {unknown}')
    if length < 1 or length > 1000:
        raise ValueError('length must be between 1 and 1000')
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    model.eval()
    input_ids = torch.tensor([[stoi[c] for c in seed_text]], dtype=torch.long)
    generated = list(seed_text)
    hidden = None
    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_ids, hidden)
            logits = output[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(itos[next_id])
            input_ids = torch.tensor([[next_id]], dtype=torch.long)
    return ''.join(generated)
