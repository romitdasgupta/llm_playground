"""
Test suite for Text Generation Playground API.
Run with: pytest test_api.py -v
"""

import pytest
import requests
import time
from typing import Generator

# Configuration
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 30  # seconds


@pytest.fixture(scope="module")
def api_client() -> Generator[requests.Session, None, None]:
    """Create a requests session for API testing."""
    session = requests.Session()

    # Wait for API to be ready
    for _ in range(10):
        try:
            response = session.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(2)
    else:
        pytest.fail("API not available after 20 seconds")

    yield session
    session.close()


class TestHealthEndpoints:
    """Test health and status endpoints."""

    def test_root_endpoint(self, api_client):
        """Test root endpoint returns basic info."""
        response = api_client.get(f"{API_BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data

    def test_health_check(self, api_client):
        """Test health check endpoint."""
        response = api_client.get(f"{API_BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] > 0
        assert len(data["available_models"]) > 0

    def test_list_models(self, api_client):
        """Test list models endpoint."""
        response = api_client.get(f"{API_BASE_URL}/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # Check model structure
        model = data[0]
        assert "name" in model
        assert "description" in model
        assert "loaded" in model

    def test_list_strategies(self, api_client):
        """Test list strategies endpoint."""
        response = api_client.get(f"{API_BASE_URL}/strategies")
        assert response.status_code == 200
        data = response.json()
        assert "strategies" in data
        assert len(data["strategies"]) == 4

        # Check strategy names
        strategy_names = [s["name"] for s in data["strategies"]]
        assert "greedy" in strategy_names
        assert "beam" in strategy_names
        assert "sampling" in strategy_names
        assert "top_k" in strategy_names


class TestTextGeneration:
    """Test text generation with different strategies."""

    def test_generate_greedy(self, api_client):
        """Test text generation with greedy decoding."""
        payload = {
            "prompt": "Once upon a time",
            "model": "gpt2",
            "strategy": "greedy",
            "max_length": 50,
        }
        response = api_client.post(
            f"{API_BASE_URL}/generate", json=payload, timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "generated_text" in data
        assert "prompt" in data
        assert "model" in data
        assert "strategy" in data
        assert "parameters" in data

        # Check values
        assert data["prompt"] == payload["prompt"]
        assert data["model"] == payload["model"]
        assert data["strategy"] == payload["strategy"]
        assert len(data["generated_text"]) > len(payload["prompt"])
        assert data["generated_text"].startswith(payload["prompt"])

    def test_generate_beam_search(self, api_client):
        """Test text generation with beam search."""
        payload = {
            "prompt": "The future of AI is",
            "model": "gpt2",
            "strategy": "beam",
            "num_beams": 5,
            "max_length": 50,
        }
        response = api_client.post(
            f"{API_BASE_URL}/generate", json=payload, timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()

        assert data["strategy"] == "beam"
        assert data["parameters"]["type"] == "beam_search"
        assert data["parameters"]["num_beams"] == 5
        assert len(data["generated_text"]) > len(payload["prompt"])

    def test_generate_sampling(self, api_client):
        """Test text generation with sampling."""
        payload = {
            "prompt": "In a galaxy far away",
            "model": "gpt2",
            "strategy": "sampling",
            "temperature": 0.8,
            "top_p": 0.95,
            "max_length": 50,
        }
        response = api_client.post(
            f"{API_BASE_URL}/generate", json=payload, timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()

        assert data["strategy"] == "sampling"
        assert data["parameters"]["type"] == "nucleus_sampling"
        assert data["parameters"]["temperature"] == 0.8
        assert data["parameters"]["top_p"] == 0.95

    def test_generate_top_k(self, api_client):
        """Test text generation with top-k sampling."""
        payload = {
            "prompt": "Hello, world!",
            "model": "gpt2",
            "strategy": "top_k",
            "temperature": 0.7,
            "top_k": 50,
            "max_length": 50,
        }
        response = api_client.post(
            f"{API_BASE_URL}/generate", json=payload, timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()

        assert data["strategy"] == "top_k"
        assert data["parameters"]["type"] == "top_k_sampling"
        assert data["parameters"]["temperature"] == 0.7
        assert data["parameters"]["top_k"] == 50

    def test_generate_with_qwen(self, api_client):
        """Test text generation with Qwen model."""
        payload = {
            "prompt": "The best way to learn programming is",
            "model": "qwen",
            "strategy": "sampling",
            "temperature": 0.8,
            "max_length": 50,
        }
        response = api_client.post(
            f"{API_BASE_URL}/generate", json=payload, timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "qwen"

    def test_generate_with_tinyllama(self, api_client):
        """Test text generation with TinyLlama Chat model."""
        payload = {
            "prompt": "Explain why the sky looks blue",
            "model": "tinyllama",
            "strategy": "sampling",
            "temperature": 0.7,
            "max_length": 50,
        }
        response = api_client.post(
            f"{API_BASE_URL}/generate", json=payload, timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "tinyllama"


class TestParameterValidation:
    """Test parameter validation and error handling."""

    def test_empty_prompt(self, api_client):
        """Test that empty prompt is rejected."""
        payload = {
            "prompt": "",
            "model": "gpt2",
            "strategy": "greedy",
            "max_length": 50,
        }
        response = api_client.post(f"{API_BASE_URL}/generate", json=payload)
        assert response.status_code == 422  # Validation error

    def test_invalid_model(self, api_client):
        """Test that invalid model is rejected."""
        payload = {
            "prompt": "Test",
            "model": "invalid_model",
            "strategy": "greedy",
            "max_length": 50,
        }
        response = api_client.post(f"{API_BASE_URL}/generate", json=payload)
        assert response.status_code == 422  # Validation error

    def test_invalid_strategy(self, api_client):
        """Test that invalid strategy is rejected."""
        payload = {
            "prompt": "Test",
            "model": "gpt2",
            "strategy": "invalid_strategy",
            "max_length": 50,
        }
        response = api_client.post(f"{API_BASE_URL}/generate", json=payload)
        assert response.status_code == 422  # Validation error

    def test_temperature_out_of_range(self, api_client):
        """Test that temperature outside valid range is rejected."""
        # Too low
        payload = {
            "prompt": "Test",
            "model": "gpt2",
            "strategy": "sampling",
            "temperature": 0.0,
            "max_length": 50,
        }
        response = api_client.post(f"{API_BASE_URL}/generate", json=payload)
        assert response.status_code == 422

        # Too high
        payload["temperature"] = 3.0
        response = api_client.post(f"{API_BASE_URL}/generate", json=payload)
        assert response.status_code == 422

    def test_max_length_out_of_range(self, api_client):
        """Test that max_length outside valid range is rejected."""
        # Too low
        payload = {
            "prompt": "Test",
            "model": "gpt2",
            "strategy": "greedy",
            "max_length": 10,
        }
        response = api_client.post(f"{API_BASE_URL}/generate", json=payload)
        assert response.status_code == 422

        # Too high
        payload["max_length"] = 1000
        response = api_client.post(f"{API_BASE_URL}/generate", json=payload)
        assert response.status_code == 422

    def test_very_long_prompt(self, api_client):
        """Test that very long prompt is rejected."""
        payload = {
            "prompt": "test " * 1000,  # Very long prompt
            "model": "gpt2",
            "strategy": "greedy",
            "max_length": 50,
        }
        response = api_client.post(f"{API_BASE_URL}/generate", json=payload)
        assert response.status_code == 422


class TestEdgeCases:
    """Test edge cases and corner cases."""

    def test_single_word_prompt(self, api_client):
        """Test generation with single word prompt."""
        payload = {
            "prompt": "Hello",
            "model": "gpt2",
            "strategy": "greedy",
            "max_length": 30,
        }
        response = api_client.post(
            f"{API_BASE_URL}/generate", json=payload, timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["generated_text"]) > len(payload["prompt"])

    def test_min_max_length(self, api_client):
        """Test generation with minimum max_length."""
        payload = {
            "prompt": "Test",
            "model": "gpt2",
            "strategy": "greedy",
            "max_length": 20,  # Minimum allowed
        }
        response = api_client.post(
            f"{API_BASE_URL}/generate", json=payload, timeout=TIMEOUT
        )
        assert response.status_code == 200

    def test_min_temperature(self, api_client):
        """Test generation with minimum temperature."""
        payload = {
            "prompt": "Test",
            "model": "gpt2",
            "strategy": "sampling",
            "temperature": 0.1,  # Minimum allowed
            "max_length": 50,
        }
        response = api_client.post(
            f"{API_BASE_URL}/generate", json=payload, timeout=TIMEOUT
        )
        assert response.status_code == 200

    def test_max_temperature(self, api_client):
        """Test generation with maximum temperature."""
        payload = {
            "prompt": "Test",
            "model": "gpt2",
            "strategy": "sampling",
            "temperature": 2.0,  # Maximum allowed
            "max_length": 50,
        }
        response = api_client.post(
            f"{API_BASE_URL}/generate", json=payload, timeout=TIMEOUT
        )
        assert response.status_code == 200

    def test_special_characters_in_prompt(self, api_client):
        """Test generation with special characters in prompt."""
        payload = {
            "prompt": "Hello! How are you? @#$%",
            "model": "gpt2",
            "strategy": "greedy",
            "max_length": 50,
        }
        response = api_client.post(
            f"{API_BASE_URL}/generate", json=payload, timeout=TIMEOUT
        )
        assert response.status_code == 200

    def test_unicode_in_prompt(self, api_client):
        """Test generation with unicode characters in prompt."""
        payload = {
            "prompt": "こんにちは 世界",  # Japanese
            "model": "gpt2",
            "strategy": "greedy",
            "max_length": 50,
        }
        response = api_client.post(
            f"{API_BASE_URL}/generate", json=payload, timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestConcurrency:
    """Test concurrent requests."""

    def test_multiple_sequential_requests(self, api_client):
        """Test multiple sequential requests."""
        for i in range(3):
            payload = {
                "prompt": f"Test {i}",
                "model": "gpt2",
                "strategy": "greedy",
                "max_length": 30,
            }
            response = api_client.post(
                f"{API_BASE_URL}/generate", json=payload, timeout=TIMEOUT
            )
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
