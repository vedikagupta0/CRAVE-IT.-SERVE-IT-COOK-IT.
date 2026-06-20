# 🍳 Crave It. Search It. Cook It.

A multimodal recipe chatbot that retrieves dish **images** with CLIP and matching **recipes** with sentence embeddings, then lets Llama-3.3-70B (via Groq) turn it all into a chef-style answer — in whatever language you ask in.

> Ask in Hindi, Spanish, French, or Arabic. The pipeline translates, retrieves, and replies in English with full context of the conversation so far.

---

## How it works

1. **Detect & translate** — `langdetect` identifies the query language; `deep-translator` normalises it to English.
2. **Image retrieval** — OpenCLIP (`ViT-B-32`) embeds the query into a shared text-image space; FAISS returns the top-k visually closest dish photos.
3. **Text retrieval** — `all-MiniLM-L6-v2` embeds the query; FAISS returns the top-5 semantically closest recipes as RAG context.
4. **Generation** — A LangChain chain feeds the retrieved context + prior chat history into Llama-3.3-70B on Groq, which returns a structured, chef-style recipe answer.
5. **History** — Each turn is stored in memory, so follow-ups like *"make that gluten-free"* inherit full context.

```
query → detect/translate → ┬─ CLIP + FAISS  → image gallery ─┐
                            └─ MiniLM + FAISS → RAG context ──┴─→ Groq LLM → response → history
```

---

## Repository structure

```
.
├── app.py                          # Entry point — runs the chatbot (CLI / interactive loop)
├── build_indexes.py                # Builds the FAISS text + image indexes from the dataset
├── download_data.py                # Downloads/prepares the recipe dataset and images
├── requirements.txt                # Python dependencies
├── crave-it-search-it-cook-it.pptx # Slide deck walking through the architecture
├── indexes/
│   ├── recipe_text_store/          # Saved FAISS index — MiniLM recipe embeddings
│   └── recipe_clip_store/          # Saved FAISS index — CLIP image embeddings
├── .gitattributes
├── .gitignore
└── LICENSE
```

---

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/crave-it-search-it-cook-it.git
cd crave-it-search-it-cook-it
pip install -r requirements.txt
```

### 2. Set your Groq API key

```bash
export GROQ_API_KEY="your-groq-api-key"
```

Get a key from [console.groq.com](https://console.groq.com).

### 3. Download the dataset

```bash
python download_data.py
```

This pulls the [Recipe Dataset with Images, Tags & Ratings](https://www.kaggle.com/datasets/seungyeonhan1/recipe-dataset-with-images-tags-and-ratings) from Kaggle. You'll need a `kaggle.json` API token configured (`~/.kaggle/kaggle.json`).

### 4. Build the indexes

```bash
python build_indexes.py
```

This embeds every recipe with MiniLM and every image with CLIP, then saves both as FAISS indexes under `indexes/`. Re-run this any time the dataset changes — the saved indexes in this repo are checked in for convenience but can be regenerated from scratch.

### 5. Run the chatbot

```bash
python app.py
```

---

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Text retrieval | `all-MiniLM-L6-v2` + FAISS | Compact (80 MB), CPU-friendly, sub-second top-5 search |
| Image retrieval | OpenCLIP `ViT-B-32` + FAISS | Shared text-image embedding space — a text query finds visually matching dishes with no captioning step |
| Translation | `langdetect` + `deep-translator` | Auto-detects language and normalises any input to English |
| LLM | Llama-3.3-70B on Groq | Sub-second 70B inference, strong structured output |
| Orchestration | LangChain (LCEL) + `InMemoryChatMessageHistory` | Composable retriever → prompt → LLM chain with multi-turn memory |
| Data | Kaggle recipe dataset | Recipes with images, tags, servings, and ratings |

---

## Example

```
You: I want a quick chocolate cake tonight
```

The bot shows the top-3 visually matching cake images, retrieves the 5 most relevant recipes, compares them, and recommends the best fit — complete with ingredients and step-by-step instructions.

---

## Notes

- The CLIP and MiniLM indexes are pre-built and saved under `indexes/` so you can run `app.py` immediately without rebuilding, as long as the dataset paths in `build_indexes.py` match your local setup.
- All chat history is in-memory only — it resets when the process restarts. Swap in a persistent `BaseChatMessageHistory` implementation if you need cross-session memory.

---

## License

See [LICENSE](./LICENSE).
