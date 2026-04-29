# Deployment Guide

This guide explains how to deploy PlantDoc to Vercel and Hugging Face Spaces.

## 📋 Prerequisites

Before deploying, ensure you have:
- Trained model checkpoint at `artifacts/modelpt/best_model.pt`
- Vector database at `data/vector_db/` (run `python -m src.rag.ingest` to create)
- `config.yaml` configuration file
- Git repository initialized

## 🚀 Deploy to Vercel

### Step 1: Install Vercel CLI

```bash
npm install -g vercel
```

### Step 2: Login to Vercel

```bash
vercel login
```

### Step 3: Deploy

```bash
vercel
```

Follow the prompts:
- Set up and deploy? → **Yes**
- Which scope? → Select your account
- Link to existing project? → **No**
- What's your project's name? → `plantdoc-ai` (or your preferred name)
- In which directory is your code located? → `./` (current directory)
- Want to modify these settings? → **No**

Vercel will automatically detect the `vercel.json` configuration and deploy your FastAPI app.

### Step 4: Set Environment Variables (if needed)

Go to your Vercel project dashboard → Settings → Environment Variables and add any required environment variables.

### Step 5: Production Deployment

```bash
vercel --prod
```

Your app will be live at `https://your-project.vercel.app`

---

## 🤗 Deploy to Hugging Face Spaces

### Option A: Using Docker (Recommended)

#### Step 1: Create a New Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Fill in:
   - **Name**: `plantdoc-ai` (or your preferred name)
   - **License**: MIT
   - **SDK**: Docker
   - **Hardware**: CPU Basic (free) or CPU Upgrade (paid for better performance)

#### Step 2: Clone the Space

```bash
git clone https://huggingface.co/spaces/your-username/plantdoc-ai
cd plantdoc-ai
```

#### Step 3: Copy Project Files

Copy all files from your local project to the Space directory:
- `app/`
- `src/`
- `config.yaml`
- `requirements.txt`
- `Dockerfile`
- `README_HF.md` (rename to `README.md`)

#### Step 4: Prepare Artifacts

Before pushing, ensure you have:
- Model checkpoint: Place in `artifacts/modelpt/best_model.pt`
- Vector database: Place in `data/vector_db/`

#### Step 5: Push to Hugging Face

```bash
git add .
git commit -m "Initial deployment"
git push
```

Hugging Face will automatically build and deploy using the Dockerfile.

### Option B: Using Python SDK

If you prefer not to use Docker, you can use the Python SDK:

#### Step 1: Create Space with Python SDK

Follow Step 1 above, but select **Python** as the SDK instead of Docker.

#### Step 2: Clone and Copy Files

Same as Option A, but you'll need a `README.md` with Python-specific configuration.

#### Step 3: Create app.py Entry Point

Create an `app.py` file in the root:

```python
from app.main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
```

#### Step 4: Push

Same as Option A.

---

## ⚙️ Important Notes

### Model and Vector Database

Both platforms require the model checkpoint and vector database to be present. Since these are in `.gitignore`, you'll need to:

1. **Temporarily remove from .gitignore** (for deployment only):
   ```bash
   # Comment out or remove these lines from .gitignore:
   # dataset/
   # artifacts/
   ```

2. **Add and commit the artifacts**:
   ```bash
   git add artifacts/modelpt/best_model.pt data/vector_db/
   git commit -m "Add model and vector db for deployment"
   ```

3. **Push to deployment platform**

4. **Restore .gitignore** after deployment:
   ```bash
   # Uncomment the lines in .gitignore
   git add .gitignore
   git commit -m "Restore gitignore"
   ```

### Alternative: Use External Storage

For production, consider:
- **Hugging Face**: Upload model to Hugging Face Hub as a repository
- **Vercel**: Use cloud storage (AWS S3, Google Cloud Storage) and download at startup

### Resource Limitations

- **Vercel Free Tier**: 100MB bundle size, 10-second function timeout
- **Hugging Face CPU Basic**: Limited RAM, may struggle with large models
- **Hugging Face CPU Upgrade**: Better for PyTorch models (recommended)

---

## 🔍 Troubleshooting

### Vercel Issues

**Error: Function execution timeout**
- Model loading may exceed 10-second limit
- Solution: Use Vercel's paid tier or consider Hugging Face Spaces

**Error: Module not found**
- Ensure `requirements.txt` includes all dependencies
- Check `api/index.py` path configuration

### Hugging Face Issues

**Build fails during pip install**
- Reduce dependencies in `requirements.txt`
- Use specific version numbers to avoid conflicts

**Model loading error**
- Ensure model checkpoint is in the correct path
- Check that `config.yaml` has correct configuration

**Out of memory**
- Upgrade to CPU Upgrade hardware
- Consider model quantization to reduce size

---

## 📊 Monitoring

### Vercel
- View logs at: `vercel logs`
- Monitor at: [vercel.com/dashboard](https://vercel.com/dashboard)

### Hugging Face
- View logs in the "Logs" tab of your Space
- Monitor resource usage in the "Settings" tab

---

## 🔄 Continuous Deployment

### Vercel with GitHub

1. Connect your GitHub repository to Vercel
2. Enable automatic deployments
3. Every push to main will trigger a new deployment

### Hugging Face with GitHub

1. Connect your GitHub repository to Hugging Face Space
2. Enable "GitHub Sync" in Space settings
3. Every push will trigger a rebuild

---

## 📝 Post-Deployment Checklist

- [ ] Test the prediction endpoint at `/predict`
- [ ] Verify the UI loads correctly at `/`
- [ ] Check model inference speed
- [ ] Test image upload functionality
- [ ] Verify RAG treatment recommendations work
- [ ] Monitor error logs for 24 hours
- [ ] Set up alerts for failures (if available)
