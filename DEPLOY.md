# Deployment Guide — RFM Customer Intelligence Engine

## Option 1: Streamlit Community Cloud (Easiest — Free)

1. **Push to GitHub:**
   ```bash
   git init
   git add app.py rfm_engine.py requirements.txt .streamlit/config.toml README.md
   git commit -m "RFM Customer Intelligence Engine"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/rfm-customer-intelligence.git
   git push -u origin main
   ```

2. **Deploy:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click **New app**
   - Select your repo, branch `main`, main file `app.py`
   - Click **Deploy**
   - Your app will be live at `https://YOUR_APP.streamlit.app`

---

## Option 2: Render (Free Tier Available)

1. Push to GitHub (same as above)

2. Go to [render.com](https://render.com) → **New** → **Web Service**

3. Connect your GitHub repo

4. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
   - **Environment:** Python 3

5. Click **Create Web Service**

---

## Option 3: Railway

1. Push to GitHub

2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub**

3. Add a `Procfile`:
   ```
   web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```

4. Railway will auto-detect Python and deploy

---

## Why Not Vercel?

Vercel is built for **serverless functions** and **static sites**. Streamlit requires a
**persistent WebSocket connection** (long-running Python process), which Vercel does not
support. The options above (Streamlit Cloud, Render, Railway) all support long-running
processes and are the correct platforms for Streamlit apps.

If you specifically need Vercel, consider converting the app to a **FastAPI + React** or
**Next.js** frontend, but that is a fundamentally different architecture.

---

## Option 4: Docker (Self-Host Anywhere)

A `Dockerfile` is included for deploying to any Docker-compatible platform
(AWS ECS, Google Cloud Run, Azure Container Apps, DigitalOcean App Platform, etc.)

```bash
docker build -t rfm-engine .
docker run -p 8501:8501 rfm-engine
```

Then visit `http://localhost:8501`
