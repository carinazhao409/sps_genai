# Helper library integration

The original helper_lib Python source files are preserved under `helper_lib/` without modification. They are reusable utilities for training, evaluation, and generation; the existing FastAPI endpoints still use the `app/` implementations.

## Local setup
Run from the `sps_genai` project root. To use helper_lib.generator and helper_lib.trainer in the local Python environment, install the optional packages if needed:

```powershell
uv add numpy matplotlib tqdm
```

The original `pyproject.toml` and `uv.lock` are intentionally unchanged so the existing Docker environment and tested endpoints are not disrupted.

## Docker
The existing Dockerfile copies `app/` and `src/`, but not `helper_lib/`. The helper library is included in this ZIP for organization and local reuse, **not installed inside the Docker image**. This is sufficient because the existing API does not import helper_lib. If future endpoints import it, add `COPY ./helper_lib /code/helper_lib` to the Dockerfile and install its optional dependencies before rebuilding.

## Run
```powershell
docker build -t sps-genai-module7 .
docker run --rm -p 8000:80 sps-genai-module7
```

Keep a backup of your previous project until you've tested all endpoints.
