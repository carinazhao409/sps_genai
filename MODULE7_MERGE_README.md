# Module 7 cumulative FastAPI project

This is your original sps_genai project with the Module 7 LSTM endpoint added. The previous `/generate`, `/embedding`, `/predict`, and `/generate-digit` endpoints are retained.

## Windows PowerShell (from this directory)

```powershell
docker build -t sps-genai-module7 .
docker run --rm -p 8000:80 sps-genai-module7
```

Open http://localhost:8000/docs and test `POST /generate_with_rnn` with:

```json
{"start_word":"a test","length":10}
```

The LSTM uses character-level text and a small demonstration corpus; the generated continuation may be nonsensical. The seed must contain only characters from the training corpus (lowercase letters and selected punctuation).

Your original Docker setup uses uv, Python 3.13 and a large spaCy model. Docker image building may take time. The Docker image has not been built in this environment; run the commands above to verify it on your Windows machine.

The previous model weights, scripts, project dependencies, and endpoints are retained. Git internals and pytest cache were excluded from this distributable ZIP. Keep your original project as a backup.
