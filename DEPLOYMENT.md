# Bitcoin FGSMA Dashboard - Deployment Guide

## Deploying to Railway

1. **Sign up at [Railway.app](https://railway.app)**

2. **Install Railway CLI** (optional):
   ```bash
   npm i -g @railway/cli
   ```

3. **Deploy via GitHub:**
   - Push your code to GitHub
   - Connect Railway to your GitHub repo
   - Railway will auto-detect Python and deploy

4. **Deploy via CLI:**
   ```bash
   railway login
   railway init
   railway up
   ```

5. **Environment Variables** (if needed):
   - Railway will automatically set `PORT`
   - No additional env vars required for this app

6. **Domain:**
   - Railway provides a free `.railway.app` domain
   - Access your dashboard at: `https://your-app.railway.app`

## Deploying to Render

1. **Sign up at [Render.com](https://render.com)**

2. **Create New Web Service:**
   - Connect your GitHub repo
   - Or use "Deploy from Git URL"

3. **Configure Service:**
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python dashboard.py`
   - **Instance Type:** Free tier works fine

4. **Deploy:**
   - Click "Create Web Service"
   - Render will build and deploy automatically

5. **Domain:**
   - Render provides a free `.onrender.com` domain
   - Access your dashboard at: `https://your-app.onrender.com`

## Local Testing

Test before deploying:

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python dashboard.py

# Test at http://localhost:5000
```

## Files Required for Deployment

- ✅ `requirements.txt` - Python dependencies
- ✅ `Procfile` - Startup command
- ✅ `runtime.txt` - Python version
- ✅ `.gitignore` - Excluded files
- ✅ `dashboard.py` - Main application
- ✅ `service-worker.js` - PWA service worker
- ✅ `static/` - PWA assets (manifest, icons)
- ✅ `fgsma_optimized.json` - Strategy parameters

## Important Notes

- **First Load:** May take 30-60 seconds to fetch Bitcoin data
- **Data Refresh:** Updates every 30 seconds when active
- **PWA:** Works offline after first load
- **Free Tier:** Both Railway and Render offer free hosting
  - Railway: 500 hours/month free
  - Render: Always-on free tier (may sleep after 15min inactivity)

## Troubleshooting

**App won't start:**
- Check logs in Railway/Render dashboard
- Verify all dependencies in `requirements.txt`

**Slow loading:**
- First load always takes longer (fetching BTC data)
- Subsequent loads use cached data

**PWA not working:**
- Ensure HTTPS is enabled (automatic on Railway/Render)
- Clear browser cache and reinstall

## Security Notes

- Dashboard is read-only (no trading functionality)
- No API keys or sensitive data required
- Safe to deploy publicly
