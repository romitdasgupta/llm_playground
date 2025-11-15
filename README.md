# Text Generation Playground

A complete web-based text generation playground with REST API backend and modern web frontend. Users can experiment with different language models (GPT-2, Qwen) and decoding strategies (Greedy, Beam Search, Sampling, Top-k).

## Architecture

```
text-generation-playground/
├── backend/
│   ├── server.py           # FastAPI backend server
│   └── requirements.txt    # Python dependencies
└── frontend/
    └── index.html         # Web interface
```

### Components

1. **Backend (FastAPI)**
   - REST API for text generation
   - Supports multiple models (GPT-2, Qwen)
   - Multiple decoding strategies
   - Health check and monitoring endpoints

2. **Frontend (HTML/CSS/JavaScript)**
   - Modern, responsive UI
   - Real-time text generation
   - Interactive controls for all parameters

## Features

### Supported Models
- **GPT-2**: OpenAI's classic autoregressive language model
- **Qwen2-0.5B-Instruct**: Efficient instruction-tuned model

### Decoding Strategies
1. **Greedy**: Deterministic, always picks highest probability token
2. **Beam Search**: Explores multiple hypotheses for better coherence
3. **Sampling**: Random sampling with temperature control (nucleus sampling)
4. **Top-k Sampling**: Samples from top k most likely tokens

### Parameters
- Temperature (0.1 - 2.0): Controls randomness
- Max Length (20 - 500): Maximum tokens to generate
- Top-k (1 - 100): Number of top tokens for top-k sampling
- Top-p (0.0 - 1.0): Cumulative probability for nucleus sampling
- Num Beams (1 - 10): Number of beams for beam search

## Quick Start

### Prerequisites
- Python 3.8+
- pip
- Modern web browser

### Installation

1. **Clone or create the project structure**
```bash
mkdir text-generation-playground
cd text-generation-playground
```

2. **(Recommended) Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. **Install backend dependencies**
```bash
cd backend
pip install -r requirements.txt
```

### Running Locally

1. **Start the backend server**
```bash
cd backend
python server.py
```

The server will start on `http://localhost:8000`

You can check the API documentation at: `http://localhost:8000/docs`

2. **Open the frontend**
```bash
cd frontend
# Option 1: Open directly in browser
open index.html  # macOS
# or just double-click index.html

# Option 2: Use a local server (recommended)
python -m http.server 8080
# Then open http://localhost:8080
```

## API Documentation

### Endpoints

#### `GET /health`
Check server health and model status

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 2,
  "available_models": ["gpt2", "qwen"]
}
```

#### `GET /models`
List all available models

#### `GET /strategies`
List all decoding strategies with descriptions

#### `POST /generate`
Generate text with specified parameters

**Request Body:**
```json
{
  "prompt": "Once upon a time",
  "model": "gpt2",
  "strategy": "sampling",
  "temperature": 0.8,
  "max_length": 100,
  "top_p": 0.95,
  "top_k": 50,
  "num_beams": 5
}
```

**Response:**
```json
{
  "generated_text": "Once upon a time in a distant galaxy...",
  "prompt": "Once upon a time",
  "model": "gpt2",
  "strategy": "sampling",
  "parameters": {
    "type": "nucleus_sampling",
    "temperature": 0.8,
    "top_p": 0.95
  }
}
```

## Deployment Options

### Option 1: Docker (Recommended for Production)

Create `backend/Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
cd backend
docker build -t text-gen-backend .
docker run -p 8000:8000 text-gen-backend
```

### Option 2: Cloud Platform (AWS, GCP, Azure)

**AWS EC2 / Google Compute Engine:**
1. Launch an instance (recommend at least 4GB RAM)
2. Install Python and dependencies
3. Run the server with uvicorn
4. Configure security group to allow port 8000
5. Use nginx as reverse proxy

**AWS Lambda / Google Cloud Functions:**
- Use Mangum adapter for FastAPI
- Note: Cold start times may be high due to model loading

### Option 3: Platform-as-a-Service

**Heroku:**
```bash
# Procfile
web: uvicorn server:app --host 0.0.0.0 --port $PORT
```

**Railway / Render:**
- Connect your GitHub repo
- Set start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`
- Configure environment variables

### Frontend Deployment

**Static Hosting (Netlify, Vercel, GitHub Pages):**
1. Update `API_BASE_URL` in `index.html` to your backend URL
2. Deploy the frontend folder

**CDN (CloudFlare, AWS CloudFront):**
- Upload to S3 bucket
- Configure CloudFront distribution
- Update CORS settings in backend

## Production Considerations

### Backend Optimizations

1. **Model Loading**
   - Pre-load models at startup (already implemented)
   - Consider model quantization for memory efficiency
   - Use GPU if available

2. **Caching**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def cached_generate(prompt, model, strategy, temp, max_len):
       # Generate text
       pass
   ```

3. **Rate Limiting**
   ```python
   from slowapi import Limiter
   from slowapi.util import get_remote_address
   
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   
   @app.post("/generate")
   @limiter.limit("10/minute")
   async def generate_text(request: Request, ...):
       # Generate text
       pass
   ```

4. **Authentication**
   ```python
   from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
   
   security = HTTPBearer()
   
   @app.post("/generate")
   async def generate_text(
       request: GenerateRequest,
       credentials: HTTPAuthorizationCredentials = Depends(security)
   ):
       # Verify token
       pass
   ```

5. **Monitoring**
   - Add Prometheus metrics
   - Log generation times
   - Track error rates

### Security

1. **CORS Configuration**
   - Restrict `allow_origins` to your frontend domain
   - Remove wildcard `*` in production

2. **Input Validation**
   - Already implemented with Pydantic
   - Add content filtering if needed

3. **API Keys**
   - Implement API key authentication
   - Use environment variables for secrets

### Scaling

1. **Horizontal Scaling**
   - Use load balancer (nginx, AWS ALB)
   - Deploy multiple backend instances
   - Share model cache across instances

2. **Vertical Scaling**
   - Increase instance size for better performance
   - Use GPU instances for faster inference

3. **Queue System**
   - Use Celery + Redis for async processing
   - Handle long-running generations

## Performance Tips

### Model Optimization
```python
# Use half precision (FP16) for faster inference
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

### Batch Processing
```python
# Generate multiple outputs at once
outputs = model.generate(
    **inputs,
    num_return_sequences=3,  # Generate 3 variants
    **gen_kwargs
)
```

## Troubleshooting

### Common Issues

1. **Models not loading**
   - Check internet connection (models download from Hugging Face)
   - Verify sufficient disk space and RAM
   - Check logs for specific errors

2. **CORS errors**
   - Verify backend CORS settings
   - Check frontend API_BASE_URL
   - Ensure backend is running

3. **Slow generation**
   - Use GPU if available
   - Reduce max_length
   - Try smaller models
   - Enable model quantization

4. **Out of memory**
   - Reduce batch size
   - Use smaller model (Qwen 0.5B)
   - Enable gradient checkpointing
   - Use CPU offloading

## Testing

### Backend Tests
```python
import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_generate():
    response = client.post("/generate", json={
        "prompt": "Hello",
        "model": "gpt2",
        "strategy": "greedy",
        "max_length": 50
    })
    assert response.status_code == 200
    assert "generated_text" in response.json()
```

### Frontend Tests
- Use browser developer tools
- Test on different browsers (Chrome, Firefox, Safari)
- Test on mobile devices
- Verify CORS and API calls

## Advanced Usage

### Custom Model Integration
```python
# Add new model to server.py
tokenizers["custom"] = AutoTokenizer.from_pretrained("your-model-name")
models["custom"] = AutoModelForCausalLM.from_pretrained("your-model-name")
```

### API Client Example
```python
import requests

def generate(prompt, model="gpt2", strategy="sampling"):
    response = requests.post(
        "http://localhost:8000/generate",
        json={
            "prompt": prompt,
            "model": model,
            "strategy": strategy,
            "temperature": 0.8,
            "max_length": 100
        }
    )
    return response.json()

result = generate("Once upon a time")
print(result["generated_text"])
```

## License

MIT License - feel free to use and modify for your needs.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review FastAPI docs: https://fastapi.tiangolo.com
- Review Transformers docs: https://huggingface.co/docs/transformers
