# PlantDoc Frontend

Modern web interface for PlantDoc AI Plant Diagnostics system.

## 🚀 Deployment to Vercel

### Step 1: Install Vercel CLI
```bash
npm install -g vercel
```

### Step 2: Login to Vercel
```bash
vercel login
```

### Step 3: Deploy
Navigate to the frontend directory and run:
```bash
cd frontend
vercel
```

Follow the prompts:
- Set up and deploy? → **Yes**
- Which scope? → Select your account
- Link to existing project? → **No**
- What's your project's name? → `plantdoc-frontend` (or your preferred name)
- In which directory is your code located? → `./` (current directory)
- Want to modify these settings? → **No**

### Step 4: Production Deployment
```bash
vercel --prod
```

Your frontend will be live at: `https://your-project.vercel.app`

## 🔧 Configuration

The frontend is configured to connect to the Hugging Face Space backend:
- **Backend API**: `https://moeenuddin01-plantdoc-ai.hf.space/v1/prediction/`

To change the backend URL, edit `js/main.js`:
```javascript
const BACKEND_URL = 'your-backend-url/v1/prediction/';
```

## 📁 Structure

```
frontend/
├── index.html      # Main HTML file
├── css/
│   └── styles.css  # Styles
├── js/
│   └── main.js     # JavaScript logic
├── assets/         # Images and icons
├── vercel.json     # Vercel configuration
└── package.json    # Project metadata
```

## 🛠 Local Development

To run the frontend locally:
```bash
cd frontend
python -m http.server 3000
```

Then open `http://localhost:3000` in your browser.

## 🌐 Features

- **Drag & Drop Image Upload**
- **Image Preview**
- **Real-time Disease Analysis**
- **Confidence Score Display**
- **AI-Powered Treatment Recommendations**
- **Responsive Design**

## 🔗 Links

- **Backend API**: https://moeenuddin01-plantdoc-ai.hf.space
- **API Documentation**: https://moeenuddin01-plantdoc-ai.hf.space/docs
- **Hugging Face Space**: https://huggingface.co/spaces/Moeenuddin01/plantdoc-ai
