"""
FastAPI backend for Text Generation Playground
Provides REST API endpoints for text generation with multiple models and strategies.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import logging
from contextlib import asynccontextmanager
import threading
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
models = {}
tokenizers = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    logger.info("Loading models...")
    
    try:
        # Load GPT-2
        logger.info("Loading GPT-2...")
        tokenizers["gpt2"] = AutoTokenizer.from_pretrained("gpt2")
        models["gpt2"] = AutoModelForCausalLM.from_pretrained("gpt2")
        
        if tokenizers["gpt2"].pad_token is None:
            tokenizers["gpt2"].pad_token = tokenizers["gpt2"].eos_token
        
        logger.info("GPT-2 loaded successfully")
        
        # Load Qwen
        logger.info("Loading Qwen...")
        tokenizers["qwen"] = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        models["qwen"] = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        
        if tokenizers["qwen"].pad_token is None:
            tokenizers["qwen"].pad_token = tokenizers["qwen"].eos_token
        
        logger.info("Qwen loaded successfully")
        logger.info("All models loaded!")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    models.clear()
    tokenizers.clear()

# Initialize FastAPI app
app = FastAPI(
    title="Text Generation Playground API",
    description="REST API for text generation with multiple models and decoding strategies",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., min_length=1, max_length=2000, description="Input text prompt")
    model: Literal["gpt2", "qwen"] = Field(default="gpt2", description="Model to use")
    strategy: Literal["greedy", "beam", "sampling", "top_k"] = Field(
        default="sampling", 
        description="Decoding strategy"
    )
    temperature: float = Field(
        default=0.8, 
        ge=0.1, 
        le=2.0, 
        description="Temperature for sampling"
    )
    max_length: int = Field(
        default=100, 
        ge=20, 
        le=500, 
        description="Maximum length of generated text"
    )
    top_k: Optional[int] = Field(
        default=50, 
        ge=1, 
        le=100, 
        description="Top-k value for top-k sampling"
    )
    top_p: Optional[float] = Field(
        default=0.95, 
        ge=0.0, 
        le=1.0, 
        description="Top-p value for nucleus sampling"
    )
    num_beams: Optional[int] = Field(
        default=5, 
        ge=1, 
        le=10, 
        description="Number of beams for beam search"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Once upon a time in a distant galaxy,",
                "model": "gpt2",
                "strategy": "sampling",
                "temperature": 0.8,
                "max_length": 100
            }
        }

class GenerateResponse(BaseModel):
    """Response model for text generation."""
    generated_text: str
    prompt: str
    model: str
    strategy: str
    parameters: dict

class ModelInfo(BaseModel):
    """Information about available models."""
    name: str
    description: str
    loaded: bool

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: int
    available_models: list[str]

# API Endpoints

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Text Generation Playground API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check if the service and models are ready."""
    return {
        "status": "healthy" if models else "loading",
        "models_loaded": len(models),
        "available_models": list(models.keys())
    }

@app.get("/models", response_model=list[ModelInfo], tags=["Models"])
async def list_models():
    """List all available models and their status."""
    model_info = [
        {
            "name": "gpt2",
            "description": "GPT-2 - OpenAI's autoregressive language model",
            "loaded": "gpt2" in models
        },
        {
            "name": "qwen",
            "description": "Qwen2-0.5B-Instruct - Efficient instruction-tuned model",
            "loaded": "qwen" in models
        }
    ]
    return model_info

@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_text(request: GenerateRequest):
    """
    Generate text using the specified model and decoding strategy.
    
    Supports multiple decoding strategies:
    - greedy: Deterministic, picks highest probability token
    - beam: Beam search for more coherent outputs
    - sampling: Random sampling with temperature control
    - top_k: Top-k sampling with temperature control
    """
    # Check if models are loaded
    if not models or request.model not in models:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{request.model}' not loaded. Available models: {list(models.keys())}"
        )
    
    try:
        # Get model and tokenizer
        model = models[request.model]
        tokenizer = tokenizers[request.model]
        
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt", padding=True)
        
        # Build generation parameters
        gen_kwargs = {
            "max_length": request.max_length,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        # Configure strategy-specific parameters
        if request.strategy == "greedy":
            gen_kwargs["do_sample"] = False
            gen_kwargs["num_beams"] = 1
            strategy_params = {"type": "greedy"}
            
        elif request.strategy == "beam":
            gen_kwargs["do_sample"] = False
            gen_kwargs["num_beams"] = request.num_beams
            gen_kwargs["early_stopping"] = True
            strategy_params = {
                "type": "beam_search",
                "num_beams": request.num_beams
            }
            
        elif request.strategy == "sampling":
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = request.temperature
            gen_kwargs["top_p"] = request.top_p
            strategy_params = {
                "type": "nucleus_sampling",
                "temperature": request.temperature,
                "top_p": request.top_p
            }
            
        elif request.strategy == "top_k":
            gen_kwargs["do_sample"] = True
            gen_kwargs["top_k"] = request.top_k
            gen_kwargs["temperature"] = request.temperature
            strategy_params = {
                "type": "top_k_sampling",
                "temperature": request.temperature,
                "top_k": request.top_k
            }
        else:
            # This should never happen due to Pydantic validation, but satisfy linter
            raise ValueError(f"Unknown strategy: {request.strategy}")
        
        # Generate text
        logger.info(f"Generating with {request.model}, strategy={request.strategy}")
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "generated_text": generated_text,
            "prompt": request.prompt,
            "model": request.model,
            "strategy": request.strategy,
            "parameters": strategy_params
        }
        
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating text: {str(e)}"
        )

@app.post("/generate/stream", tags=["Generation"])
async def generate_text_stream(request: GenerateRequest):
    """
    Generate text with token-by-token streaming.
    
    Returns tokens as they are generated using Server-Sent Events (SSE).
    Each event contains a JSON object with the token and metadata.
    """
    # Check if models are loaded
    if not models or request.model not in models:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{request.model}' not loaded. Available models: {list(models.keys())}"
        )
    
    def generate_stream():
        try:
            # Get model and tokenizer
            model = models[request.model]
            tokenizer = tokenizers[request.model]
            
            # Tokenize input
            inputs = tokenizer(request.prompt, return_tensors="pt", padding=True)
            
            # Build generation parameters
            gen_kwargs = {
                "max_new_tokens": request.max_length - inputs.input_ids.shape[1],
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            
            # Configure strategy-specific parameters
            if request.strategy == "greedy":
                gen_kwargs["do_sample"] = False
                gen_kwargs["num_beams"] = 1
                strategy_params = {"type": "greedy"}
                
            elif request.strategy == "beam":
                gen_kwargs["do_sample"] = False
                gen_kwargs["num_beams"] = request.num_beams
                gen_kwargs["early_stopping"] = True
                strategy_params = {
                    "type": "beam_search",
                    "num_beams": request.num_beams
                }
                
            elif request.strategy == "sampling":
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = request.temperature
                gen_kwargs["top_p"] = request.top_p
                strategy_params = {
                    "type": "nucleus_sampling",
                    "temperature": request.temperature,
                    "top_p": request.top_p
                }
                
            elif request.strategy == "top_k":
                gen_kwargs["do_sample"] = True
                gen_kwargs["top_k"] = request.top_k
                gen_kwargs["temperature"] = request.temperature
                strategy_params = {
                    "type": "top_k_sampling",
                    "temperature": request.temperature,
                    "top_k": request.top_k
                }
            else:
                # This should never happen due to Pydantic validation, but satisfy linter
                raise ValueError(f"Unknown strategy: {request.strategy}")
            
            # Create streamer
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            # Start generation in a separate thread
            generation_kwargs = {**inputs, **gen_kwargs, "streamer": streamer}
            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Send initial metadata
            yield f"data: {json.dumps({'type': 'start', 'prompt': request.prompt, 'model': request.model, 'strategy': request.strategy, 'parameters': strategy_params})}\n\n"
            
            # Stream tokens as they're generated
            for token in streamer:
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
            
            thread.join()
            
        except Exception as e:
            logger.error(f"Error during streaming generation: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/strategies", tags=["Generation"])
async def list_strategies():
    """List all available decoding strategies with descriptions."""
    return {
        "strategies": [
            {
                "name": "greedy",
                "description": "Greedy decoding - always picks the highest probability token",
                "deterministic": True,
                "uses_temperature": False
            },
            {
                "name": "beam",
                "description": "Beam search - explores multiple hypotheses for better coherence",
                "deterministic": True,
                "uses_temperature": False,
                "parameters": ["num_beams"]
            },
            {
                "name": "sampling",
                "description": "Nucleus (top-p) sampling - samples from probability distribution",
                "deterministic": False,
                "uses_temperature": True,
                "parameters": ["temperature", "top_p"]
            },
            {
                "name": "top_k",
                "description": "Top-k sampling - samples from top k most likely tokens",
                "deterministic": False,
                "uses_temperature": True,
                "parameters": ["temperature", "top_k"]
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)