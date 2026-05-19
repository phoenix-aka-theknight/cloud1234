# Deploying Digit Recognition App to Render

## Pre-Deployment Steps

### 1. Prepare Your Repository
```bash
# Make sure you have git initialized
git init
git add .
git commit -m "Initial commit: Digit recognition app"
```

### 2. Important: Include the Model File
Since the model training can timeout on Render's free tier, you should:

**Option A (Recommended):** Include your pre-trained model
```bash
# Make sure model/digit_model.h5 exists locally
# Remove digit_model.h5 from .gitignore (comment it out)
# OR create model directory with the trained model
git add model/digit_model.h5
git commit -m "Add pre-trained model"
```

**Option B:** Let it train on first deploy (takes ~5-10 minutes)
- Keep current setup. Model will train on first startup.
- Note: This uses significant compute time on Render's free tier.

### 3. Push to GitHub
```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

## Deployment on Render

### Steps:

1. **Go to [render.com](https://render.com)**
   - Sign up with GitHub account
   - Authorize Render to access your repositories

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Select your repository
   - Connect the repo

3. **Configure Service**
   - **Name**: `digit-recognition` (or your choice)
   - **Runtime**: Python 3.11
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Plan**: Free (for testing) or Starter (production)
   - **Region**: Choose closest to you

4. **Click "Create Web Service"**
   - Render will build and deploy your app
   - Takes 2-5 minutes (or 10-15 if training model)

5. **View Your App**
   - After deployment, get your URL from Render dashboard
   - Example: `https://digit-recognition.onrender.com`

## Important Notes

### ⚠️ Render Free Tier Limitations
- **Startup timeout**: 30 minutes (should be fine for training)
- **Memory**: 512 MB
- **CPU**: Shared
- **Cold starts**: App may sleep after 15 minutes of inactivity
- **Disk**: Ephemeral (files not persisted between restarts)

### 🔧 Optimization Tips

**If model training keeps timing out:**
1. Reduce epochs in `app.py` (change from 15 to 5-10)
2. Reduce batch size (change from 128 to 64)
3. Pre-train locally and include `model/digit_model.h5` in repo

**For better performance:**
- Upgrade to Starter plan ($7/month) for persistent disk
- Use `tensorflow-cpu` (included in requirements.txt) instead of full TensorFlow

### 📝 Modify app.py for Production (Optional)

If you want to skip model training on Render and use a pre-downloaded model:

```python
# Add this to app.py after imports (around line 39)
import urllib.request

def download_pretrained_model():
    """Download a pre-trained model if local one doesn't exist"""
    if not os.path.exists(MODEL_PATH):
        logger.info("Downloading pre-trained model...")
        os.makedirs("model", exist_ok=True)
        # Download from your hosted location
        # urllib.request.urlretrieve(url, MODEL_PATH)
```

## Troubleshooting

### ❌ "Build failed"
- Check Python version (3.11 recommended)
- Verify `requirements.txt` syntax
- Check that `Procfile` exists and is correct

### ❌ "Application failed to bind"
- Make sure `gunicorn` is in `requirements.txt`
- Check that `Procfile` has correct format

### ❌ "Model not found / Training timeout"
- Add `model/digit_model.h5` to repository
- Remove or comment out `*.h5` in `.gitignore`

### ❌ "Out of memory"
- Reduce training data in `app.py`
- Use Starter plan for more resources

## Deployment Success!

Once deployed, you can:
- View logs: Render dashboard → Web Service → Logs
- Redeploy: Push new commits to main branch
- Test: Use the provided URL to access your drawing pad

Your app is now live! 🚀
