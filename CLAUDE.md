# CLAUDE.md
This file provides guidance to Claude Code and the Context Engineer when working with the Agros-Vision-RAG repository.

# Project Overview
A multi-modal Agentic AI system that bridges Computer Vision (ResNet-50) with Context Engineering (RAG). It identifies plant diseases from images and retrieves authoritative treatment protocols from a PDF-based knowledge base (CABI/FAO).

# Context Engineering Philosophy
As per the Vizuara Bootcamp methodology:

Long-Lived Context: Stored in cloud.md. Contains project philosophy, global constraints, and the "Central Brain" logic.

Short-Lived Context: Managed in task.md. Tracks immediate sub-tasks (e.g., Phase 1a: Ingestion) and is cleared once the state changes.

Isolate & Select: The src/ logic is modular to allow the agent to fetch only relevant code chunks into the prompt.

# Common Commands
python -m venv .venv

source .venv/bin/activate
pip install -r requirements.txt
# Ensure Ollama is running for the RAG component
ollama serve
ollama pull llama3

# Run the application
python -m src.rag.ingest
python -m src.vision.train
# Train the model
python -m src.orchestrator.main


streamlit run app/streamlit.py
uvicorn app.main:app --reload

## Code Style — PEP 8

All source files must comply with PEP 8. `ruff` is the enforcer; CI will fail on lint errors.

### Naming
- Modules and packages: `snake_case`
- Classes: `PascalCase`
- Functions and variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private helpers: prefix with a single underscore

### Docstrings
Every public function, method, and class must have a Google-style docstring. Every docstring must include `Args`, `Returns`, and `Raises` sections where applicable. Every parameter in `Args` must have a description, not just a type. The `Raises` section must list every exception the function can raise intentionally.

### Type Annotations
All function signatures must include type annotations for every parameter and the return type. Use `from __future__ import annotations` at the top of each file.

### Line Length
Maximum line length is 88 characters. Break long function signatures across multiple lines with one argument per line.

### File / Module Size
Keep each source file between 300 and 500 lines. If a file grows beyond 500 lines, split it into focused sub-modules and re-export the public API from the package `__init__.py`.

### Imports
Order imports in three blocks separated by a blank line: standard library, third-party packages, then internal project modules.

### Other Rules
- Prefer `pathlib.Path` over raw string paths.
- Never use bare `except:`; always catch a specific exception type.
- Avoid mutable default arguments; use `None` and assign inside the function body.
- Use f-strings for string formatting; avoid `%` and `.format()`.

---

## FastAPI Best Practices

### Project Structure
Keep all API concerns inside `app/`. Organise into `routers/` (one file per domain), `schemas/` (separate files for request and response models), `dependencies.py` (shared `Depends()` providers), and `middleware.py` (CORS, logging, error handlers).

### Application Factory
Define the app via a `create_app()` factory function rather than a bare module-level instance. This keeps the app testable and configurable. Register all routers and middleware inside the factory.

### Pydantic Schemas
Define explicit Pydantic models for every request and response body; never use raw `dict`. Every field must have a type annotation and a `Field(description=...)` value so that the auto-generated OpenAPI docs are meaningful.

### Dependency Injection
Load heavy resources such as the model and label encoder once at startup using `Depends()` with `@lru_cache`. Never instantiate or load models inside a route handler.

### Route Handlers
Route handlers must be thin: validate input, call a service function from `src/`, and return a schema. Business logic must live in `src/`, not in `app/routers/`. Every route must declare a `summary`, `response_model`, and explicit `status_code`.

### Error Handling
Register a global exception handler in `middleware.py`. Do not wrap individual route handlers in try/except blocks for general errors.

### Async vs Sync
Use `async def` for I/O-bound routes. Use plain `def` for CPU-bound routes so FastAPI runs them in a thread pool. Never call blocking code directly inside an `async def` without offloading to an executor.

### Versioning
Prefix all routes with `/v1/` from day one. Add a `/v2/` router for breaking changes rather than modifying existing routes.

---

## Architecture

### `src/` — Core library
- **`src/data/loader.py`** — Dataset loading; wraps PyTorch `Dataset`/`DataLoader` for raw images from `data/raw/`
- **`src/data/encoder.py`** — Label encoding for skin lesion classes
- **`src/model/loader.py`** — Loads/instantiates ResNet-152; handles pretrained weights and checkpoint saving/loading
- **`src/model/train.py`** — Training loop (forward pass, loss, optimizer step)
- **`src/model/evaluation.py`** — Metrics computation (accuracy, confusion matrix, etc.)
- **`src/model/prediction.py`** — Inference on single images or batches
- **`src/utils.py`** — Shared utilities (device selection, path helpers)

### `src/pipelines/` — Orchestration scripts
- **`data_preprocessing.py`** — Reads `data/raw/`, applies transforms, writes to `data/processed/`
- **`model_training.py`** — Runs full training: data loading → model setup → train loop → checkpoint save
- **`model_evaluation.py`** — Loads a checkpoint and evaluates on a test split

### `app/`
- **`main.py`** — Application factory and entry point
- **`routers/`** — Route handlers grouped by domain
- **`schemas/`** — Pydantic request/response models
- **`dependencies.py`** — Shared `Depends()` providers (model, encoder)
- **`middleware.py`** — CORS, logging, global error handlers
- **`streamlit.py`** — Interactive demo; uploads an image and returns the predicted lesion class

### `data/`
- `data/raw/` — Original dataset (not tracked by git)
- `data/processed/` — Preprocessed/augmented data ready for training (not tracked by git)

### `notebooks/`
Jupyter notebooks for exploratory work; not part of the production pipeline.

---


 # Package,Role
torch / torchvision,Vision Backbone (ResNet-50)
langchain / langchain-community,RAG Orchestration
chromadb,Vector Database (Short/Long lived retrieval)
ollama,Local Open-Source LLM (Llama-3)
fastapi,RESTful Service
streamlit,Interactive UI for Teacher Demos






## Application Layer Structure Rule

All machine learning projects must include an `app/` directory responsible for serving the trained model through an API and user interface.

The `app/` directory must follow this structure:

app/
├── main.py
├── dependencies.py
├── middleware.py
├── model_loader.py
├── predict.py
├── schemas.py
└── templates/
└── index.html

### Responsibilities of Each File

**main.py**

* Entry point of the API service.
* Initializes the FastAPI application.
* Registers routes and middleware.

**dependencies.py**

* Contains reusable FastAPI dependency functions.
* Handles shared resources like model loading or configuration injection.

**middleware.py**

* Contains custom middleware such as logging, request tracking, or CORS configuration.

**model_loader.py**

* Responsible for loading the trained machine learning model.
* Must not perform training.
* Only loads model weights from the artifacts directory.

**predict.py**

* Contains the prediction logic.
* Receives input data, preprocesses it if necessary, and returns model predictions.

**schemas.py**

* Defines request and response schemas using Pydantic models.

**templates/**

* Contains HTML templates for the web interface.
* Used when serving a simple UI alongside the API.

### Architectural Rules

1. The `app/` directory must only contain **serving and inference logic**.
2. No model training code is allowed inside `app/`.
3. Training code must remain in `src/model/` or `src/pipelines/`.
4. The `model_loader.py` file must load models from `artifacts/`.
5. API routes must be defined in `main.py` and call functions from `predict.py`.

### Purpose

This separation ensures that:

* Training logic remains in `src/`
* Deployment logic remains in `app/`
* The project stays modular and production-ready.
