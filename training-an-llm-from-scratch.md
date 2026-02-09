# Training an LLM from Scratch with Minimal Bias for Ollama

This guide covers building a language model from the ground up -- not fine-tuning someone else's model, but training your own from a blank slate. This gives you full control over what the model learns, which is the most direct path to controlling bias.

**Be aware:** Training from scratch is orders of magnitude harder and more expensive than fine-tuning. A useful model requires large amounts of data, significant compute, and patience. The steps below are real and actionable, but the results from a home setup will be modest compared to commercial models. That said, a small, purpose-built model trained on carefully chosen data can be genuinely useful for specific tasks.

---

## What You Will Need

- **A computer with a modern NVIDIA GPU** (24GB+ VRAM strongly recommended; e.g., RTX 3090, RTX 4090, or an A100 via cloud rental)
- **Hundreds of gigabytes of free disk space** for storing training data
- **Python 3.10+** installed on your system
- **Time** -- training can take days or weeks depending on model size and hardware
- Alternatively, **cloud GPU rental** from services like vast.ai, Lambda Labs, or RunPod (expect to spend $1-5/hour for capable hardware)

---

## Step 1: Define Your Model's Purpose and Scope

Before collecting any data, decide:

1. **What is this model for?** A general-purpose chatbot? A domain-specific assistant (legal, medical, coding)? A creative writing tool?
2. **What biases are you specifically trying to avoid?** Political slant? Cultural assumptions? Gender stereotypes? Write these down explicitly.
3. **How large a model can you realistically train?** On a single consumer GPU, you are looking at models in the range of 100M to 1B parameters. Anything above 1B requires multi-GPU setups or cloud compute.

Writing this down first will guide every decision that follows.

---

## Step 2: Acquire Training Data

Your training data is the single biggest factor in both the quality and the bias of your model. You need a large, diverse text corpus.

### Free and Open Data Sources

| Source | What It Contains | How to Get It | Bias Considerations |
|--------|-----------------|---------------|-------------------|
| **Wikipedia dumps** | Encyclopedia articles in many languages | [dumps.wikimedia.org](https://dumps.wikimedia.org) -- download the XML dump for your language | Skews toward Western/English-speaking perspectives; better coverage of some topics than others |
| **Project Gutenberg** | 70,000+ public domain books | [gutenberg.org](https://www.gutenberg.org) | Heavily weighted toward older Western literature; reflects historical attitudes |
| **Common Crawl** | Petabytes of raw web pages | [commoncrawl.org](https://commoncrawl.org) | Contains everything on the web, including toxic and biased content; requires heavy filtering |
| **The Pile** (by EleutherAI) | 800GB curated English text dataset | [Hugging Face](https://huggingface.co/datasets/EleutherAI/the_pile) | Already filtered and organized into subsets; well-documented biases |
| **OSCAR** | Multilingual web text corpus | [oscar-project.org](https://oscar-project.github.io/documentation/) | Good for non-English data; web-sourced so still needs filtering |
| **RedPajama** | Open reproduction of LLaMA training data | [Hugging Face](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) | 1.2 trillion tokens; well-documented composition |
| **Stack Exchange data dump** | Q&A across hundreds of topics | [archive.org/details/stackexchange](https://archive.org/details/stackexchange) | Community-moderated; skews toward technical topics |
| **OpenSubtitles** | Movie and TV subtitles in many languages | [opus.nlpl.eu](https://opus.nlpl.eu/OpenSubtitles-v2018.php) | Reflects Hollywood/entertainment biases; good for conversational style |
| **Academic papers** | Research across all fields | [Semantic Scholar](https://www.semanticscholar.org/product/api) or [arXiv bulk access](https://arxiv.org/help/bulk_data) | Formal language; reflects academic institutional biases |
| **Government records** | Laws, proceedings, public data | Various .gov open data portals | Bureaucratic language; reflects government perspectives |
| **CulturaX** | 6.3 trillion tokens, 167 languages | [Hugging Face](https://huggingface.co/datasets/uonlp/CulturaX) | One of the most linguistically diverse open datasets |

### How Much Data Do You Need?

- **100M parameter model:** ~2-5 billion tokens minimum (roughly 8-20 GB of raw text)
- **500M parameter model:** ~10-20 billion tokens (roughly 40-80 GB of raw text)
- **1B parameter model:** ~20-50 billion tokens (roughly 80-200 GB of raw text)

A "token" is roughly 3/4 of a word in English.

### Downloading the Data

Most of these sources provide download scripts or direct download links. For example:

- **Wikipedia:** Download the latest dump file (e.g., `enwiki-latest-pages-articles.xml.bz2`) and use a tool called **WikiExtractor** (free on GitHub) to pull out the plain text.
- **The Pile / RedPajama:** Download directly from Hugging Face using their command-line tool (`huggingface-cli download`).
- **Common Crawl:** Use their provided index to download specific segments (downloading the whole thing is impractical -- it is petabytes).

---

## Step 3: Clean and Filter Your Data

Raw data is messy. This step directly impacts bias.

### What to Remove
1. **Duplicate content** -- repeated text teaches the model to repeat, not to think. Use a deduplication tool like **MinHash** (available in the `datasketch` Python library) or **ExactSubstr Deduplication** from EleutherAI.
2. **Toxic and hateful content** -- use a toxicity classifier to score and remove highly toxic text. Free options include the **Detoxify** Python library or **Perspective API** (free for research use from Google/Jigsaw).
3. **Spam, ads, and boilerplate** -- web-sourced data is full of cookie notices, navigation menus, and ads. Tools like **trafilatura** (Python library) extract article text from HTML and discard the junk.
4. **Extremely low-quality text** -- gibberish, encoding errors, auto-generated content. Filter by perplexity score using a small pre-existing language model as a quality judge.

### What to Balance
1. **Topic distribution** -- count how much text you have per topic/domain. If 80% of your data is tech content and 2% is arts and humanities, your model will reflect that imbalance.
2. **Source diversity** -- track which sources contribute what percentage of your data. No single source should dominate.
3. **Language and cultural representation** -- if you want a multilingual or culturally aware model, ensure non-English and non-Western sources are meaningfully represented, not token inclusions.
4. **Time period** -- mixing old and modern text is fine, but be aware that older texts may contain outdated or offensive viewpoints. You may want to down-weight very old content or annotate it.

### Document Everything

Create a data card (a simple text document) that records:
- Every source you used
- How much data came from each source
- What you filtered out and why
- What known biases remain
- The date you assembled the dataset

This is not just good practice -- it is the only way to meaningfully talk about your model's biases later.

---

## Step 4: Train a Tokenizer

A tokenizer is the component that breaks text into small pieces (tokens) the model can process. Training your own ensures it fits your specific data.

**Tool: SentencePiece** (free, by Google) or **Hugging Face Tokenizers library**

1. Install: `pip install sentencepiece` or `pip install tokenizers`
2. Feed it a representative sample of your training data (a few gigabytes is enough)
3. Choose a vocabulary size:
   - **32,000 tokens** is a common default and works well for English
   - **50,000-64,000 tokens** is better if you have multilingual data
4. The tool outputs a tokenizer model file you will use in the next step

With the Hugging Face tokenizers library, training a BPE (Byte-Pair Encoding) tokenizer looks roughly like:
```
- Install the tokenizers library
- Point it at your text files
- Set the vocabulary size
- Run the training
- Save the result
```

Unsloth also provides notebook-based workflows for this if you prefer a visual interface.

---

## Step 5: Choose a Model Architecture

You are not inventing a new architecture -- you are choosing an existing, proven design and setting its size.

**Recommended architecture: Transformer decoder-only** (same family as GPT, LLaMA, Mistral)

You need to decide on these numbers:

| Parameter | Small (100M) | Medium (500M) | Large (1B) |
|-----------|-------------|---------------|------------|
| Layers | 12 | 24 | 24-32 |
| Hidden size | 768 | 1024 | 2048 |
| Attention heads | 12 | 16 | 16-32 |
| Context length | 1024-2048 | 2048 | 2048-4096 |

Start small. A 100M-parameter model trains in hours to days on a single GPU and lets you validate your entire pipeline before committing to a larger run.

---

## Step 6: Train the Model

### Recommended Tools (Free and Open Source)

- **LitGPT** ([github.com/Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt)) -- the most user-friendly option for training GPT-style models from scratch. Provides clear config files and scripts.
- **NanoGPT** ([github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)) -- extremely simple and educational; good for smaller models and learning the process.
- **GPT-NeoX** (by EleutherAI) -- designed for larger-scale training across multiple GPUs.
- **Hugging Face Transformers** -- the most flexible but also the most complex to configure for from-scratch training.

### The Training Process (using LitGPT as an example)

1. **Install LitGPT:** `pip install litgpt`
2. **Prepare your data:** LitGPT provides a `prepare` command that tokenizes your text files using your tokenizer and converts them into the binary format the trainer expects.
3. **Write a config file:** This specifies your model architecture (from Step 5), learning rate, batch size, and how long to train. LitGPT provides example configs you can modify.
4. **Start training:** Run the training command. It will read your prepared data, train the model, and save checkpoints periodically.
5. **Monitor progress:** Watch the "loss" number. It should decrease over time. If it stops decreasing, training is done (or you need more/better data).

### Practical Tips

- **Save checkpoints frequently** -- if training crashes (power outage, GPU error), you can resume from the last checkpoint instead of starting over.
- **Start with a small model first** -- train a 100M model to validate your data pipeline works before spending days on a larger one.
- **Watch for overfitting** -- if the model starts memorizing your training data word-for-word instead of learning patterns, you need more data or a smaller model.
- **Use mixed precision (FP16 or BF16)** -- this cuts memory usage roughly in half and speeds up training. Most modern tools enable this by default.

---

## Step 7: Convert the Trained Model to GGUF Format

Ollama requires models in GGUF format. Use **llama.cpp** to convert your model.

1. **Clone llama.cpp:** `git clone https://github.com/ggerganov/llama.cpp`
2. **Install its Python requirements:** `pip install -r llama.cpp/requirements.txt`
3. **Export your model to Hugging Face format** (if not already) -- LitGPT and most training tools can export to the standard Hugging Face `transformers` format.
4. **Run the conversion script:**
   ```
   python llama.cpp/convert_hf_to_gguf.py /path/to/your/model --outtype f16
   ```
5. **Quantize the model** to reduce file size and memory usage:
   ```
   ./llama.cpp/llama-quantize your-model.gguf your-model-Q5_K_M.gguf Q5_K_M
   ```

Quantization options:
- **Q8_0** -- best quality, largest file
- **Q5_K_M** -- good balance of quality and size
- **Q4_K_M** -- smallest usable size, some quality loss

---

## Step 8: Load into Ollama

1. **Install Ollama** from [ollama.com](https://ollama.com)
2. **Create a Modelfile** (a plain text file):
   ```
   FROM ./your-model-Q5_K_M.gguf

   PARAMETER temperature 0.7
   PARAMETER top_p 0.9

   TEMPLATE """{{ .Prompt }}"""
   ```
3. **Create the model in Ollama:**
   ```
   ollama create my-model -f Modelfile
   ```
4. **Run it:**
   ```
   ollama run my-model
   ```

Your model is now running locally.

---

## Step 9: Evaluate for Bias

Your model is trained, but you need to test it.

1. **Prompt it with sensitive topics** -- politics, religion, race, gender, nationality. Does it show a clear lean?
2. **Ask parallel questions** -- "Tell me about [Group A]" vs. "Tell me about [Group B]." Are the responses comparable in tone and depth?
3. **Use established benchmarks:**
   - **BBQ (Bias Benchmark for QA)** -- tests for social biases across many categories
   - **WinoBias** -- tests for gender bias in pronoun resolution
   - **CrowS-Pairs** -- tests for stereotypical associations
   - These are all freely available on Hugging Face.
4. **Have diverse testers** -- people from different backgrounds will notice biases you will not.
5. **Compare against your data card** from Step 3 -- do the biases you documented in your data show up in the model's outputs?

If the results are unacceptable, return to Step 3 (adjust your data) and retrain.

---

## Step 10: Iterate

A first attempt will not be perfect. The cycle is:

1. Train
2. Evaluate
3. Identify problems
4. Adjust data (add underrepresented sources, remove problematic content, rebalance)
5. Retrain
6. Re-evaluate

Each cycle gets you closer to a model that reflects the values and balance you are aiming for.

---

## Realistic Expectations

| Model Size | Single GPU Training Time | Quality Level |
|-----------|------------------------|---------------|
| 100M params | Hours to 1-2 days | Can complete sentences and follow simple patterns; useful for learning the process |
| 500M params | 3-7 days | Can produce coherent paragraphs; useful for narrow, domain-specific tasks |
| 1B params | 1-3 weeks | Can hold short conversations and follow instructions with additional alignment training |
| 7B+ params | Weeks to months; multi-GPU required | Approaching the quality floor of models people use daily |

Commercial models like ChatGPT and Claude are trained on clusters of thousands of GPUs over months. A single-GPU home project will not match that. But a small model trained on carefully chosen data can outperform a large model on specific topics where your data is better than theirs -- and it will be *yours*, with biases you chose and documented rather than ones chosen for you.
