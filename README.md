# llama1B-ORPO

**A Colab notebook that fine-tunes Llama 3.2 1B Instruct with ORPO preference optimization, exports it to GGUF and runs it locally with Ollama.**

> **Context.** Built in February 2026 as a personal learning project, when I was finishing my Computer Engineering degree at UPV and working at Grupo CEU. It was my first pass at LLM fine-tuning, a topic I kept working on in later projects. It is a one-notebook tutorial, not a tuned or evaluated model.

## What it does

- Loads `unsloth/Llama-3.2-1B-Instruct` in 4-bit and attaches LoRA adapters (r=16, rsLoRA).
- Fine-tunes it with ORPO on a 5,000-example sample of `trl-lib/ultrafeedback_binarized`.
- Exports the merged model to a single F16 GGUF file and generates an Ollama `Modelfile`.
- Copies both files to Google Drive so they survive the Colab session.
- Includes explanations of ORPO, LoRA, 4-bit quantization and GGUF in the notebook's markdown cells.

## How it works

1. Install Unsloth, TRL and Datasets.
2. Load the base model with `FastLanguageModel` (max sequence length 1024, 4-bit).
3. Convert UltraFeedback pairs to `{prompt, chosen, rejected}` and drop examples over 8,000 characters.
4. Train with `ORPOTrainer`: learning rate 5e-6, beta 0.1, batch size 2 with 4 gradient accumulation steps (effective 8), 200 steps.
5. Call `save_pretrained_gguf(..., quantization_method="f16")`, which merges the adapters and converts with llama.cpp.
6. Create the Ollama model from the generated GGUF and `Modelfile`.

## Results

Printed outputs saved in `llama1B.ipynb` (one run on a Colab Tesla T4, 14.7 GB):

- 4,875 examples left after the length filter.
- 11,272,192 of 1,247,086,592 parameters trained (0.90%).
- 200 steps, runtime 970.7 s, training loss 1.873 (`TrainOutput`).
- The 200 steps cover about 0.33 of an epoch of the filtered set.

No benchmark or comparison against the base model was run. The single test prompt in the notebook (asking for a one-sentence explanation of ORPO) got an answer that describes ORPO incorrectly, so the notebook shows that the pipeline works end to end, not that the model improved.

## Run it

1. Open `llama1B.ipynb` in Google Colab, or locally with a CUDA GPU (the notebook says 8 GB of VRAM is enough). Choose a GPU runtime and run the cells in order. The Unsloth install can take several minutes. The Drive cell needs Colab; skip it elsewhere.
2. Download `llama-3.2-1b-instruct.F16.gguf` and `Modelfile` from the Drive folder `ollama_export_llama1b`. They are not in the repo because of their size.
3. Install [Ollama](https://ollama.com) and, in the folder with both files:

```bash
ollama create llama1b-orpo -f Modelfile
ollama run llama1b-orpo
```

I did not re-run the notebook or the Ollama steps when writing this README.

## Repository layout

```
llama1B.ipynb   training, GGUF export and test, with explanations
README.md       this file
```

## Stack

Python, Unsloth, TRL (ORPOTrainer), Hugging Face Datasets, LoRA, llama.cpp (GGUF), Ollama, Google Colab.

## Model license

The repo has no license file. The base model is subject to Meta's Llama 3.2 license.
