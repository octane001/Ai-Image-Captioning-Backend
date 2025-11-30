# AI Image Captioning Backend 🤖🖼️

A FastAPI-based REST API that leverages the BLIP (Bootstrapping Language-Image Pre-training) transformer model to generate natural language descriptions of images, designed to power accessible applications for visually impaired individuals.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

## ✨ Features

- 🧠 **BLIP Transformer Model** - State-of-the-art vision-language model
- ⚡ **Fast Inference** - GPU acceleration support
- 🔄 **Batch Processing** - Handle multiple images simultaneously
- 📝 **Detailed Captions** - Standard and detailed description modes
- 🎯 **RESTful API** - Clean, well-documented endpoints
- 🔒 **CORS Enabled** - Ready for frontend integration
- 📊 **Interactive Docs** - Auto-generated Swagger UI

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- 4GB+ RAM (8GB recommended)
- (Optional) CUDA-capable GPU for faster processing

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-image-captioning-backend.git
cd ai-image-captioning-backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
# Start the API server
python main.py

# Server will start at http://localhost:8000
# Access interactive docs at http://localhost:8000/docs
```

## 📋 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "BLIP",
  "device": "cuda"
}
```

---

#### 2. Generate Caption
```http
POST /caption
```

**Parameters:**
- `file` (required) - Image file (JPG, PNG, etc.)
- `max_length` (optional, default: 50) - Maximum caption length
- `detailed` (optional, default: false) - Generate detailed description

**Request Example:**
```python
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/caption',
        files={'file': f},
        data={'detailed': 'true'}
    )
    print(response.json())
```

**Response:**
```json
{
  "success": true,
  "caption": "a dog sitting on grass in a park",
  "alternative_captions": [
    "a golden retriever relaxing on green lawn",
    "a pet dog enjoying outdoor time"
  ],
  "image_size": "1920x1080",
  "detailed": true
}
```

---

#### 3. Batch Caption
```http
POST /batch-caption
```

**Parameters:**
- `files` (required) - Multiple image files (max 10)

**Response:**
```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "caption": "a cat sleeping on a couch",
      "success": true
    },
    {
      "filename": "image2.jpg",
      "caption": "a sunset over mountains",
      "success": true
    }
  ]
}
```

---

#### 4. Root Endpoint
```http
GET /
```

**Response:**
```json
{
  "message": "Image Captioning API for Accessibility",
  "endpoints": {
    "/caption": "POST - Upload image for captioning",
    "/health": "GET - Check API health"
  }
}
```

## 🛠️ Tech Stack

- **Framework**: FastAPI 0.104+
- **ML Model**: BLIP (Salesforce Research)
- **Deep Learning**: PyTorch 2.1+
- **Transformers**: Hugging Face Transformers
- **Image Processing**: Pillow (PIL)
- **Server**: Uvicorn

## 📁 Project Structure

```
ai-image-captioning-backend/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── README.md           # Documentation
└── .gitignore          # Git ignore file
```

## 🔧 Configuration

### Model Selection

The default model is BLIP base. To use different models, modify `main.py`:

```python
# BLIP-2 (more accurate, larger model)
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
```

### GPU Configuration

To force CPU usage:
```python
device = "cpu"
model.to(device)
```

To use specific GPU:
```python
device = "cuda:0"  # Use first GPU
model.to(device)
```

### CORS Settings

Modify allowed origins in `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Server Configuration

Change host and port:
```python
uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 📦 Requirements

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
transformers==4.35.2
torch==2.1.1
torchvision==0.16.1
pillow==10.1.0
python-multipart==0.0.6
accelerate==0.25.0
```

## 🚀 Performance Tips

### GPU Acceleration
Install CUDA toolkit for 5-10x faster inference:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Model Caching
Models are automatically cached after first download (~1GB):
- **Linux/Mac**: `~/.cache/huggingface/`
- **Windows**: `C:\Users\USERNAME\.cache\huggingface\`

### Optimization
- Use batch processing for multiple images
- Resize large images before upload
- Consider using BLIP-base for faster inference
- Enable GPU for production deployments

## 🐛 Troubleshooting

### Model Download Issues
```bash
# Manually download model
python -c "from transformers import BlipProcessor, BlipForConditionalGeneration; \
BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base'); \
BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')"
```

### Out of Memory Errors
- Reduce batch size
- Use BLIP-base instead of BLIP-2
- Close other applications
- Force CPU mode if GPU memory is limited

### CORS Errors
- Verify frontend URL in CORS settings
- Check if both servers are running
- Clear browser cache

### Port Already in Use
```bash
# Change port in main.py or kill existing process
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -i :8000
kill -9 <PID>
```

## 🧪 Testing

### Test with cURL
```bash
curl -X POST "http://localhost:8000/caption" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

### Test with Python
```python
import requests

url = "http://localhost:8000/caption"
files = {'file': open('test_image.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

### Interactive Testing
Visit `http://localhost:8000/docs` for Swagger UI with interactive API testing.

## 🌐 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t image-captioning-api .
docker run -p 8000:8000 image-captioning-api
```

### Production Considerations

1. **Use Production ASGI Server**
```python
# Use gunicorn with uvicorn workers
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

2. **Add Authentication**
```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/caption")
async def generate_caption(token: str = Depends(security)):
    # Verify token
    pass
```

3. **Rate Limiting**
```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/caption")
@limiter.limit("10/minute")
async def generate_caption():
    pass
```

4. **Logging**
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## 🔐 Security

- Validate file types before processing
- Implement rate limiting
- Add authentication for production
- Sanitize file uploads
- Use HTTPS in production
- Set proper CORS origins

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/Enhancement`)
3. Commit your changes (`git commit -m 'Add Enhancement'`)
4. Push to the branch (`git push origin feature/Enhancement`)
5. Open a Pull Request

### Development Setup
```bash
# Install dev dependencies
pip install pytest black flake8

# Run tests
pytest

# Format code
black main.py

# Lint code
flake8 main.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [BLIP Model](https://github.com/salesforce/BLIP) by Salesforce Research
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [FastAPI Framework](https://fastapi.tiangolo.com/)
- [PyTorch](https://pytorch.org/)

## 📚 References

- [BLIP Paper](https://arxiv.org/abs/2201.12086)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

## 📞 Support

For issues, questions, or feature requests:
- Open an [Issue](https://github.com/yourusername/ai-image-captioning-backend/issues)
- Email: devpiyushkumar870@gmail.com
- Documentation: Check `/docs` endpoint

## 🔄 Changelog

### v1.0.0 (2024)
- Initial release
- BLIP model integration
- FastAPI REST API
- Batch processing support
- Detailed caption mode

---

**Built with ❤️ for accessibility and AI innovation**
