# Training a Custom LLM with Minimal Bias for Ollama

This is a simplified guide. "Training" here really means **fine-tuning** an existing open-source model on your own curated data -- training one from scratch requires millions of dollars in compute. Fine-tuning is the realistic path for an individual.

---

## Step 1: Pick a Base Model

Go to [huggingface.co](https://huggingface.co) and choose a pre-trained open-source model. Good starting points:

- **Llama 3** (by Meta) -- strong general-purpose model
- **Mistral** or **Mixtral** -- efficient and capable
- **Gemma** (by Google) -- smaller, runs on modest hardware

Pick a size your computer can handle. A model with **7B or 8B parameters** is a reasonable starting point (needs ~16GB RAM minimum, a GPU with 16GB+ VRAM helps enormously). Smaller models (1B-3B) work on weaker machines.

---

## Step 2: Curate Your Training Data (This Is Where Bias Lives)

This is the most important step for reducing bias. Your data determines what the model learns.

**Where to get data (free sources):**
- **Wikipedia** -- broad, community-edited, relatively neutral
- **Project Gutenberg** -- public domain books from many eras and perspectives
- **Common Crawl** -- a massive open web archive (needs heavy filtering)
- **Hugging Face Datasets** -- search for curated, pre-cleaned datasets
- **Academic papers** via Semantic Scholar or arXiv (for factual grounding)
- **Government open data** portals (census, public records)

**How to reduce bias in your data:**
1. **Use multiple sources** -- don't rely on one website, one viewpoint, or one culture
2. **Check representation** -- does your data include diverse perspectives (geographic, cultural, demographic)?
3. **Remove overtly toxic content** -- filter out hate speech, slurs, and extremist material
4. **Balance topics** -- don't let one subject dominate (e.g., 90% politics, 10% everything else)
5. **Read samples** -- manually spot-check your data for obvious slant or repetition
6. **Document your choices** -- write down what you included and excluded, and why

**Format your data** as a simple JSONL file (one JSON object per line):
```
{"instruction": "What is photosynthesis?", "output": "Photosynthesis is the process by which plants convert sunlight..."}
```

There are free tools to help with formatting, such as Python scripts on Hugging Face or simple spreadsheet-to-JSONL converters.

---

## Step 3: Fine-Tune the Model

The most accessible free tool for this is **Unsloth** ([github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)). It is designed to make fine-tuning fast and memory-efficient, even on a single consumer GPU.

**What you need:**
- A computer with an NVIDIA GPU (12GB+ VRAM recommended), **or**
- A free **Google Colab** notebook (Unsloth provides ready-made Colab notebooks you can run in your browser -- no local GPU needed)

**The process (using Unsloth's Colab notebook):**
1. Open one of Unsloth's provided Google Colab notebooks
2. Change the model name to the base model you chose in Step 1
3. Upload your JSONL dataset file
4. Point the notebook to your dataset
5. Click "Run All" and wait for training to finish
6. Save the result

Unsloth walks you through each cell in the notebook with comments. You are mostly clicking "play" buttons and changing a few text fields.

---

## Step 4: Export the Model to GGUF Format

Ollama uses the **GGUF** file format. Unsloth can export directly to GGUF -- there is a cell in the notebook for this. You choose a **quantization level**, which controls the tradeoff between quality and file size:

- **Q8_0** -- highest quality, largest file
- **Q5_K_M** -- good balance (recommended starting point)
- **Q4_K_M** -- smaller file, slightly lower quality

This gives you a single `.gguf` file.

---

## Step 5: Import into Ollama

1. **Install Ollama** from [ollama.com](https://ollama.com) (free, works on Mac, Windows, Linux)
2. Create a file called `Modelfile` with this content:
   ```
   FROM ./your-model-name.Q5_K_M.gguf
   PARAMETER temperature 0.7
   SYSTEM "You are a helpful assistant."
   ```
3. Open a terminal and run:
   ```
   ollama create my-custom-model -f Modelfile
   ```
4. Run your model:
   ```
   ollama run my-custom-model
   ```

---

## Step 6: Test for Bias

After your model is running, test it:

1. **Ask the same question from different angles** -- e.g., ask about a political topic and see if it favors one side
2. **Ask about different demographic groups** -- does it describe them differently or unfairly?
3. **Ask controversial questions** -- does it present multiple viewpoints or just one?
4. **Use evaluation datasets** -- Hugging Face has bias-evaluation datasets like BBQ and WinoBias you can run through your model
5. **Have other people test it** -- your own blind spots are, by definition, invisible to you

If you find problems, go back to Step 2, adjust your data, and re-run the fine-tuning.

---

## Key Realities to Keep in Mind

- **Bias cannot be fully eliminated.** Every dataset reflects the world it came from. The goal is to *reduce and be aware of* bias, not to achieve zero bias.
- **The base model already has biases** baked into it from its original training. Fine-tuning adjusts behavior but does not erase the foundation.
- **More diverse data helps more than more data.** 10,000 well-chosen examples from varied sources beat 1,000,000 examples from one source.
- **Fine-tuning is not the same as training from scratch.** You are steering an existing model, not building one from nothing. This is both a strength (much cheaper and easier) and a limitation (you inherit the base model's assumptions).
- **"Unbiased" is itself a subjective concept.** Be explicit about what biases you are trying to reduce and document your methodology so others can evaluate your choices.
