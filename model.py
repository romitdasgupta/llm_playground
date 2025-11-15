import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load both GPT-2 and Qwen models using HuggingFace `.from_pretrained` method.
"""
YOUR CODE HERE (~10-15 lines of code)
"""
gpt2_model_name = "gpt2"
gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name)
gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_model_name)

qwen_model_name = "Qwen/Qwen3-0.6B"
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
qwen_model = AutoModelForCausalLM.from_pretrained(qwen_model_name)


gpt2_model.eval()
qwen_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2_model.to(device)
qwen_model.to(device)

print("\nModel architectures:")
print(f"GPT-2 parameters: {sum(p.numel() for p in gpt2_model.parameters()):,}")
print(f"Qwen parameters: {sum(p.numel() for p in qwen_model.parameters()):,}")


def generate(model, tokenizer, prompt, strategy="greedy", max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if strategy == "greedy":
        gen_kwargs = {
            "do_sample": False,
        }
    elif strategy == "top-k":
        gen_kwargs = {
            "do_sample": True,
            "top_k": 50,
            "temperature": 0.7,
        }
    elif strategy == "top-p":
        gen_kwargs = {
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.7,
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            **gen_kwargs,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)
