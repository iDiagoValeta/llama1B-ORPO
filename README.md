# 🦙 Fine-Tune Llama 3.2 1B with ORPO & Deploy with Ollama

A complete **pedagogical tutorial** for fine-tuning Llama 3.2 1B using ORPO (Odds Ratio Preference Optimization) and deploying it locally with Ollama.

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `llama1B.ipynb` | Complete training notebook with step-by-step explanations |
| `README.md` | This guide |

> ⚠️ **Note**: The GGUF model file and Modelfile are **not included** due to their large size (~2.5GB). You'll generate them by running the notebook.

---

## 🎯 What You'll Learn

1. **Fine-tune an LLM** using ORPO preference optimization
2. **Export to GGUF format** for efficient local inference
3. **Deploy with Ollama** for easy local usage
4. **Understand LoRA, quantization, and preference learning**

---

## 🚀 Getting Started

### Step 1: Run the Training Notebook

Open `llama1B.ipynb` in **Google Colab** (recommended) or locally with a CUDA GPU:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iDiagoValeta/llama1B-ORPO/blob/master/llama1B.ipynb)

The notebook will:
- Fine-tune Llama 3.2 1B with ORPO on the UltraFeedback dataset
- Export the model as a `.gguf` file
- Generate a `Modelfile` for Ollama

### Step 2: Install Ollama

Download from [ollama.ai](https://ollama.ai):
- **Windows**: Download and run the installer
- **macOS**: `brew install ollama`
- **Linux**: `curl -fsSL https://ollama.ai/install.sh | sh`

Verify installation:
```bash
ollama --version
```

### Step 3: Create Your Ollama Model

After running the notebook, you'll have two files:
- `llama-3.2-1b-instruct.F16.gguf` (the fine-tuned model)
- `Modelfile` (Ollama configuration)

Navigate to the folder containing these files and run:

```bash
ollama create llama1b-orpo -f Modelfile
```

**Expected output:**
```
transferring model data
creating model layer
writing manifest
success
```

### Step 4: Run Your Model

```bash
ollama run llama1b-orpo
```

You'll enter an interactive chat:
```
>>> Hello! What can you help me with?
```

---

## 💻 Using Your Model in Applications

#### Python Example
```python
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama1b-orpo",
        "prompt": "Explain machine learning in simple terms.",
        "stream": False
    }
)
print(response.json()["response"])
```

#### cURL Example
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama1b-orpo",
  "prompt": "What is ORPO?",
  "stream": false
}'
```

#### Using the OpenAI-Compatible API
Ollama provides an OpenAI-compatible endpoint:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Can be any string
)

response = client.chat.completions.create(
    model="llama1b-orpo",
    messages=[
        {"role": "user", "content": "Hello! How are you?"}
    ]
)
print(response.choices[0].message.content)
```

---

## 📖 Understanding the Modelfile

The notebook generates a `Modelfile` that tells Ollama how to run your model:

```dockerfile
FROM llama-3.2-1b-instruct.F16.gguf

TEMPLATE """{{ if .Messages }}..."""

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
```

| Component | Purpose |
|-----------|---------|
| `FROM` | Path to the GGUF model file |
| `TEMPLATE` | Chat formatting template (Llama 3 format) |
| `PARAMETER stop` | Tokens that signal end of generation |
| `PARAMETER temperature` | Creativity level (higher = more creative) |
| `PARAMETER min_p` | Minimum probability threshold for sampling |

### Customizing Your Modelfile

You can adjust the model behavior by modifying the `Modelfile`:

```dockerfile
# More deterministic responses
PARAMETER temperature 0.7

# Longer responses
PARAMETER num_predict 512

# Add a system prompt
SYSTEM "You are a helpful assistant specialized in explaining AI concepts."
```

After modifying, recreate the model:
```bash
ollama create llama1b-orpo -f Modelfile
```

---

## 🧠 About the Training Process

This model was fine-tuned using **ORPO (Odds Ratio Preference Optimization)**, a technique that:

1. **Combines SFT and alignment** in a single training phase
2. **Eliminates the reference model** required by DPO/RLHF
3. **Reduces memory usage** and training time

### Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | Llama 3.2 1B Instruct |
| Method | ORPO with LoRA (r=16) |
| Dataset | UltraFeedback (5,000 examples) |
| Training Steps | 200 |
| Quantization | F16 (GGUF) |

For the complete training code, see `llama1B.ipynb`.

---

## 💡 Tips and Troubleshooting

### Common Issues

**"Model not found" error:**
```bash
# Make sure you're in the correct directory
cd /path/to/folder/with/gguf/and/Modelfile
ollama create llama1b-orpo -f Modelfile
```

**Model runs slowly:**
- Ensure Ollama is using your GPU: check with `ollama ps`
- For CPU-only systems, consider using a smaller quantization (q4_k_m)

**Out of memory:**
- The F16 model requires ~2-3GB RAM
- Use a quantized version for lower memory usage

### Checking Model Status

```bash
# List all local models
ollama list

# Show model information
ollama show llama1b-orpo

# Remove a model
ollama rm llama1b-orpo
```

### Running as a Service

Ollama runs as a background service. To check its status:

```bash
# Check if Ollama is running
ollama ps

# Start Ollama service (if not auto-started)
ollama serve
```

---

## 🔄 Customizing the Training

Want to train on your own data? Modify the notebook:

### Changing the Dataset

Replace the dataset loading code:
```python
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
```

With your own dataset in this format:
```python
# Each example needs:
{
    "prompt": "User's question or instruction",
    "chosen": "Good response (preferred)",
    "rejected": "Bad response (not preferred)"
}
```

## 📚 Resources

- **ORPO Paper**: [arXiv:2403.07691](https://arxiv.org/abs/2403.07691)
- **Ollama Documentation**: [ollama.ai](https://ollama.ai)
- **Unsloth** (Fast Fine-tuning): [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)
- **TRL Library**: [github.com/huggingface/trl](https://github.com/huggingface/trl)
- **GGUF Format**: [github.com/ggerganov/ggml](https://github.com/ggerganov/ggml)

---

## 📄 License

The model weights are subject to Meta's Llama license. Please review:
- [Llama 3.2 License](https://ai.meta.com/llama/license/)

---

## 🤝 Contributing

Found an issue or want to improve this guide? Feel free to open an issue or pull request!
