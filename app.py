"""
Crave It · Search It · Cook It
Multimodal Recipe Chatbot — Gradio + LangChain + CLIP + Groq
"""

import os
import io
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from PIL import Image
import torch
import open_clip
import gradio as gr

from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── Config ───────────────────────────────────────────────────────────────────

GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
DATA_PATH      = os.environ.get("DATA_PATH", "data/recipes_images.json")
IMAGE_FOLDER   = os.environ.get("IMAGE_FOLDER", "data/images")
TEXT_INDEX_DIR = "indexes/recipe_text_store"
CLIP_INDEX_DIR = "indexes/recipe_clip_store"
TOP_K_IMAGES   = 3
TOP_K_TEXT     = 5
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Device: {DEVICE}")

# ── Translation ───────────────────────────────────────────────────────────────

def translate_to_english(text: str) -> str:
    try:
        lang = detect(text)
        if lang == "en":
            return text
        return GoogleTranslator(source=lang, target="en").translate(text)
    except (LangDetectException, Exception):
        return text

# ── CLIP Embeddings Wrapper ───────────────────────────────────────────────────

class OpenCLIPEmbeddings(Embeddings):
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device="cpu"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def embed_documents(self, image_paths: list[str]) -> list[list[float]]:
        embeddings = []
        with torch.no_grad():
            for path in image_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    img_t = self.preprocess(img).unsqueeze(0).to(self.device)
                    feat = self.model.encode_image(img_t)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    embeddings.append(feat.cpu().numpy()[0].tolist())
                except Exception as e:
                    logger.warning(f"Skipping image {path}: {e}")
                    embeddings.append([0.0] * 512)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            feat = self.model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()[0].tolist()

# ── RAG text builder ──────────────────────────────────────────────────────────

def build_rag_column(df: pd.DataFrame, cols: list[str], new_col="rag_text") -> pd.DataFrame:
    def concat_row(row):
        parts = []
        for c in cols:
            val = row[c]
            if isinstance(val, list):
                val = ", ".join(map(str, val))
            parts.append(f"{c}: {val}")
        return " ;; ".join(parts)
    df[new_col] = df.apply(concat_row, axis=1)
    df[new_col] = df[new_col].str.replace("\\", "", regex=False)
    return df

# ── Index builder / loader ────────────────────────────────────────────────────

def build_or_load_indexes():
    text_emb  = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    clip_emb  = OpenCLIPEmbeddings(device=DEVICE)

    # ── Text index ──
    if os.path.exists(TEXT_INDEX_DIR):
        logger.info("Loading existing text index …")
        vs_text = FAISS.load_local(TEXT_INDEX_DIR, text_emb, allow_dangerous_deserialization=True)
    else:
        logger.info("Building text index …")
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. See README.")
        df = pd.read_json(DATA_PATH)
        df["servings"].fillna("4-5", inplace=True)
        df["ratings"].fillna({"rating": 0.0, "count": 0}, inplace=True)
        df["description"].fillna("Description unavailable.", inplace=True)
        df["image_filename"].fillna("", inplace=True)
        df = build_rag_column(df, list(df.columns))
        docs = [
            Document(page_content=row["rag_text"], metadata={"recipe_id": idx + 1})
            for idx, row in df.iterrows()
        ]
        os.makedirs(TEXT_INDEX_DIR, exist_ok=True)
        vs_text = FAISS.from_documents(docs, text_emb)
        vs_text.save_local(TEXT_INDEX_DIR)
        logger.info(f"Text index saved — {len(docs):,} docs")

    # ── CLIP index ──
    if os.path.exists(CLIP_INDEX_DIR):
        logger.info("Loading existing CLIP index …")
        vs_clip = FAISS.load_local(CLIP_INDEX_DIR, clip_emb, allow_dangerous_deserialization=True)
    else:
        logger.info("Building CLIP index …")
        if not os.path.exists(IMAGE_FOLDER):
            logger.warning("Image folder not found — CLIP index will be empty.")
            vs_clip = None
        else:
            valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
            image_paths = [
                os.path.join(IMAGE_FOLDER, f)
                for f in os.listdir(IMAGE_FOLDER)
                if os.path.splitext(f)[1].lower() in valid_ext
            ]
            image_docs = [
                Document(page_content=path, metadata={"image_path": path})
                for path in image_paths
            ]
            os.makedirs(CLIP_INDEX_DIR, exist_ok=True)
            vs_clip = FAISS.from_documents(image_docs, clip_emb)
            vs_clip.save_local(CLIP_INDEX_DIR)
            logger.info(f"CLIP index saved — {len(image_docs):,} images")

    return vs_text, vs_clip

# ── LangChain chain ───────────────────────────────────────────────────────────

PROMPT = ChatPromptTemplate.from_template("""\
You are an expert chef and recipe assistant.

Answer ONLY using the provided recipe context. If multiple recipes are relevant:
- Compare them briefly
- Explain the key differences
- Recommend the best match
- Present the full winning recipe cleanly with ingredients and steps

If the context contains no relevant recipe, say so honestly.

---
Recipe context:
{context}

Conversation history:
{chat_history}

Question:
{question}
""")

def build_chain(vs_text, history):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Add it in Space secrets.")
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0)
    retriever = vs_text.as_retriever(search_kwargs={"k": TOP_K_TEXT})
    chain = (
        {
            "chat_history": RunnableLambda(lambda _: history.messages),
            "context": retriever,
            "question": lambda x: x,
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain

# ── Image helpers ─────────────────────────────────────────────────────────────

def clip_gallery_images(query: str, vs_clip, k=TOP_K_IMAGES) -> list[tuple[Image.Image, str]]:
    """Return (PIL Image, caption) pairs for the top-k CLIP results."""
    if vs_clip is None:
        return []
    try:
        docs = vs_clip.similarity_search(query, k=k)
    except Exception as e:
        logger.warning(f"CLIP search error: {e}")
        return []
    gallery = []
    for doc in docs:
        path = doc.metadata.get("image_path", "")
        try:
            img = Image.open(path).convert("RGB")
            caption = os.path.basename(path).replace("-", " ").replace("_", " ").split(".")[0].title()
            gallery.append((img, caption))
        except Exception:
            pass
    return gallery

# ── Gradio state init ─────────────────────────────────────────────────────────

logger.info("Initialising indexes …")
vs_text, vs_clip = build_or_load_indexes()
history = InMemoryChatMessageHistory()
chain   = build_chain(vs_text, history)
logger.info("Ready.")

# ── Chat callback ─────────────────────────────────────────────────────────────

def respond(user_msg: str, chat_history: list):
    if not user_msg.strip():
        return chat_history, [], ""

    # Translate
    english = translate_to_english(user_msg)
    translated_note = f"*(Translated: {english})*\n\n" if english != user_msg else ""

    # CLIP gallery
    gallery_imgs = clip_gallery_images(english, vs_clip)

    # RAG answer
    try:
        answer = chain.invoke(english)
    except Exception as e:
        answer = f"⚠️ Error: {e}"

    history.add_user_message(english)
    history.add_ai_message(answer)

    full_answer = translated_note + answer
    chat_history = chat_history + [[user_msg, full_answer]]
    return chat_history, gallery_imgs, ""

def clear_session():
    history.clear()
    return [], [], ""

# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
#chatbox { height: 520px; overflow-y: auto; }
.gallery-row { margin-top: 8px; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="🍳 Crave It · Search It · Cook It") as demo:
    gr.Markdown(
        """
# 🍳 Crave It · Search It · Cook It
**Multimodal Recipe Chatbot** — semantic text retrieval · CLIP image search · Llama-3.3-70B on Groq

Ask anything in **any language** — the bot translates, finds the closest recipes visually and semantically, then gives you a chef-style answer.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(elem_id="chatbox", label="Chat")
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask a recipe question (e.g. 'chocolate lava cake', 'मसाला दाल', 'pasta carbonara') …",
                    show_label=False,
                    scale=9,
                )
                send_btn = gr.Button("Send 🍴", variant="primary", scale=1)
            clear_btn = gr.Button("🗑️ Clear session", variant="secondary")

        with gr.Column(scale=2):
            gr.Markdown("### 🖼️ Visually Similar Dishes")
            gallery = gr.Gallery(
                label="CLIP image matches",
                columns=3,
                height=320,
                object_fit="cover",
                elem_classes=["gallery-row"],
            )

    gr.Markdown(
        """
---
**Dataset:** [Recipe Dataset with Images, Tags & Ratings](https://www.kaggle.com/datasets/seungyeonhan1/recipe-dataset-with-images-tags-and-ratings) · 
**Stack:** LangChain · FAISS · OpenCLIP ViT-B-32 · MiniLM-L6-v2 · Groq Llama-3.3-70B
        """
    )

    # Wiring
    send_btn.click(respond, [msg_box, chatbot], [chatbot, gallery, msg_box])
    msg_box.submit(respond, [msg_box, chatbot], [chatbot, gallery, msg_box])
    clear_btn.click(clear_session, [], [chatbot, gallery, msg_box])

if __name__ == "__main__":
    demo.launch()
