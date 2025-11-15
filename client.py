"""
Python client library for Text Generation Playground API.
Provides a simple interface to interact with the text generation backend.
"""

import requests
from typing import Literal, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """Result of text generation."""
    generated_text: str
    prompt: str
    model: str
    strategy: str
    parameters: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict) -> 'GenerationResult':
        """Create GenerationResult from API response."""
        return cls(
            generated_text=data['generated_text'],
            prompt=data['prompt'],
            model=data['model'],
            strategy=data['strategy'],
            parameters=data['parameters']
        )


class TextGenerationClient:
    """Client for Text Generation Playground API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.

        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def health_check(self) -> Dict[str, Any]:
        """
        Check server health and model status.

        Returns:
            Health check response with status and available models

        Raises:
            requests.RequestException: If the request fails
        """
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def list_models(self) -> list:
        """
        List all available models.

        Returns:
            List of model information dictionaries

        Raises:
            requests.RequestException: If the request fails
        """
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()

    def list_strategies(self) -> Dict[str, Any]:
        """
        List all available decoding strategies.

        Returns:
            Dictionary containing strategy information

        Raises:
            requests.RequestException: If the request fails
        """
        response = self.session.get(f"{self.base_url}/strategies")
        response.raise_for_status()
        return response.json()

    def generate(
        self,
        prompt: str,
        model: Literal["gpt2", "qwen"] = "gpt2",
        strategy: Literal["greedy", "beam", "sampling", "top_k"] = "sampling",
        temperature: float = 0.8,
        max_length: int = 100,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        num_beams: Optional[int] = 5
    ) -> GenerationResult:
        """
        Generate text using the specified parameters.

        Args:
            prompt: Input text prompt
            model: Model to use ('gpt2' or 'qwen')
            strategy: Decoding strategy ('greedy', 'beam', 'sampling', 'top_k')
            temperature: Temperature for sampling (0.1 to 2.0)
            max_length: Maximum length of generated text (20 to 500)
            top_k: Top-k value for top-k sampling (1 to 100)
            top_p: Top-p value for nucleus sampling (0.0 to 1.0)
            num_beams: Number of beams for beam search (1 to 10)

        Returns:
            GenerationResult containing the generated text and metadata

        Raises:
            ValueError: If parameters are invalid
            requests.RequestException: If the request fails
        """
        # Validate parameters
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if not 0.1 <= temperature <= 2.0:
            raise ValueError("Temperature must be between 0.1 and 2.0")

        if not 20 <= max_length <= 500:
            raise ValueError("Max length must be between 20 and 500")

        # Prepare request payload
        payload = {
            "prompt": prompt,
            "model": model,
            "strategy": strategy,
            "temperature": temperature,
            "max_length": max_length
        }

        # Add optional parameters
        if top_k is not None:
            payload["top_k"] = top_k
        if top_p is not None:
            payload["top_p"] = top_p
        if num_beams is not None:
            payload["num_beams"] = num_beams

        # Make request
        response = self.session.post(
            f"{self.base_url}/generate",
            json=payload
        )
        response.raise_for_status()

        # Parse and return result
        return GenerationResult.from_dict(response.json())

    def generate_greedy(
        self,
        prompt: str,
        model: Literal["gpt2", "qwen"] = "gpt2",
        max_length: int = 100
    ) -> GenerationResult:
        """
        Generate text using greedy decoding.

        Args:
            prompt: Input text prompt
            model: Model to use
            max_length: Maximum length of generated text

        Returns:
            GenerationResult containing the generated text
        """
        return self.generate(
            prompt=prompt,
            model=model,
            strategy="greedy",
            max_length=max_length
        )

    def generate_beam(
        self,
        prompt: str,
        model: Literal["gpt2", "qwen"] = "gpt2",
        num_beams: int = 5,
        max_length: int = 100
    ) -> GenerationResult:
        """
        Generate text using beam search.

        Args:
            prompt: Input text prompt
            model: Model to use
            num_beams: Number of beams
            max_length: Maximum length of generated text

        Returns:
            GenerationResult containing the generated text
        """
        return self.generate(
            prompt=prompt,
            model=model,
            strategy="beam",
            num_beams=num_beams,
            max_length=max_length
        )

    def generate_sampling(
        self,
        prompt: str,
        model: Literal["gpt2", "qwen"] = "gpt2",
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_length: int = 100
    ) -> GenerationResult:
        """
        Generate text using nucleus (top-p) sampling.

        Args:
            prompt: Input text prompt
            model: Model to use
            temperature: Temperature for sampling
            top_p: Cumulative probability for nucleus sampling
            max_length: Maximum length of generated text

        Returns:
            GenerationResult containing the generated text
        """
        return self.generate(
            prompt=prompt,
            model=model,
            strategy="sampling",
            temperature=temperature,
            top_p=top_p,
            max_length=max_length
        )

    def generate_top_k(
        self,
        prompt: str,
        model: Literal["gpt2", "qwen"] = "gpt2",
        temperature: float = 0.8,
        top_k: int = 50,
        max_length: int = 100
    ) -> GenerationResult:
        """
        Generate text using top-k sampling.

        Args:
            prompt: Input text prompt
            model: Model to use
            temperature: Temperature for sampling
            top_k: Number of top tokens to sample from
            max_length: Maximum length of generated text

        Returns:
            GenerationResult containing the generated text
        """
        return self.generate(
            prompt=prompt,
            model=model,
            strategy="top_k",
            temperature=temperature,
            top_k=top_k,
            max_length=max_length
        )

    def close(self):
        """Close the session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Example usage
if __name__ == "__main__":
    # Create client
    client = TextGenerationClient("http://localhost:8000")

    # Check health
    try:
        health = client.health_check()
        print(f"Server status: {health['status']}")
        print(f"Available models: {health['available_models']}")
    except requests.RequestException as e:
        print(f"Error: {e}")
        exit(1)

    # Example 1: Simple generation with defaults
    print("\n--- Example 1: Sampling ---")
    result = client.generate(
        prompt="Once upon a time in a distant galaxy,",
        model="gpt2",
        strategy="sampling",
        temperature=0.7
    )
    print(f"Prompt: {result.prompt}")
    print(f"Generated: {result.generated_text}")

    # Example 2: Greedy decoding
    print("\n--- Example 2: Greedy ---")
    result = client.generate_greedy(
        prompt="The future of artificial intelligence is",
        max_length=80
    )
    print(f"Generated: {result.generated_text}")

    # Example 3: Beam search
    print("\n--- Example 3: Beam Search ---")
    result = client.generate_beam(
        prompt="In the year 2050,",
        num_beams=5,
        max_length=100
    )
    print(f"Generated: {result.generated_text}")

    # Example 4: Top-k sampling
    print("\n--- Example 4: Top-k Sampling ---")
    result = client.generate_top_k(
        prompt="The most important discovery in physics was",
        temperature=0.9,
        top_k=40
    )
    print(f"Generated: {result.generated_text}")

    # Example 5: Using context manager
    print("\n--- Example 5: Context Manager ---")
    with TextGenerationClient() as client:
        result = client.generate("Hello, world!")
        print(f"Generated: {result.generated_text}")

    