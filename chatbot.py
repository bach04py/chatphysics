import os
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant

from qdrant_client import QdrantClient

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

import torch
import numpy as np

# =========================
# ENV
# =========================
load_dotenv()

DATA_PATH = "./data/physics.pdf"
QDRANT_PATH = "./qdrant_db"

# =========================
# 4BIT QUANT MODEL
# =========================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.2
)

# =========================
# EMBEDDING
# =========================
EMBEDDING = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

RERANK_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# QDRANT
# =========================
def load_db():
    client = QdrantClient(path=QDRANT_PATH)

    try:
        db = Qdrant(client=client, collection_name="physics", embedding=EMBEDDING)
        db.similarity_search("test", k=1)
        return db, []
    except:
        pass

    docs = PyPDFLoader(DATA_PATH).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(docs)

    db = Qdrant.from_documents(
        docs,
        EMBEDDING,
        client=client,
        collection_name="physics"
    )

    return db, docs

# =========================
# HYBRID RETRIEVER
# =========================
class HybridRetriever:

    def __init__(self, docs, db):
        self.docs = docs
        self.db = db
        self.bm25 = BM25Okapi([d.page_content.split() for d in docs]) if docs else None

    def retrieve(self, query, k=4):
        vec_docs = self.db.similarity_search(query, k=k)

        if self.bm25:
            scores = self.bm25.get_scores(query.split())
            idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            bm_docs = [self.docs[i] for i in idx]
            vec_docs.extend(bm_docs)

        return vec_docs[:k]

# =========================
# RERANK
# =========================
def rerank(query, docs, k=2):

    q_vec = RERANK_MODEL.encode([query])[0]

    scored = []
    for d in docs:
        d_vec = RERANK_MODEL.encode([d.page_content])[0]
        scored.append((np.dot(q_vec, d_vec), d))

    scored.sort(reverse=True)

    return [d for _, d in scored[:k]]

# =========================
# TOOLS
# =========================
def academic_solver(q):
    if "lab" in q.lower():
        return "No. Lab = 0 → fail."
    return None

def physics_solver(q):
    if "resistance" in q.lower():
        return "R = r/3"
    return None

# =========================
# STATE
# =========================
class State(TypedDict, total=False):
    query: str
    route: str
    context: str
    answer: str
    prompt: str

# =========================
# NODES
# =========================
def planner(state):
    q = state["query"].lower()

    if "lab" in q:
        return {**state, "route": "academic"}
    if "resistance" in q:
        return {**state, "route": "physics"}

    return {**state, "route": "rag"}

def academic_node(state):
    return {**state, "answer": academic_solver(state["query"])}

def physics_node(state):
    return {**state, "answer": physics_solver(state["query"])}

def rag_node(state):
    docs = RETRIEVER.retrieve(state["query"])
    docs = rerank(state["query"], docs)

    context = "\n".join([d.page_content[:200] for d in docs])
    return {**state, "context": context}

def llm_node(state):
    prompt = f"""
Context:
{state.get("context","")}

Question:
{state["query"]}

Answer clearly step-by-step:
"""
    return {**state, "prompt": prompt}

# =========================
# GRAPH
# =========================
def build_graph():
    g = StateGraph(State)

    g.add_node("planner", planner)
    g.add_node("academic", academic_node)
    g.add_node("physics", physics_node)
    g.add_node("rag", rag_node)
    g.add_node("llm", llm_node)

    g.set_entry_point("planner")

    g.add_conditional_edges(
        "planner",
        lambda s: s["route"],
        {
            "academic": "academic",
            "physics": "physics",
            "rag": "rag"
        }
    )

    g.add_edge("rag", "llm")
    g.add_edge("academic", END)
    g.add_edge("physics", END)
    g.add_edge("llm", END)

    return g.compile()

# =========================
# CHATBOT
# =========================
class PhysicsChatbot:

    def __init__(self):
        global DB, RETRIEVER
        DB, docs = load_db()
        RETRIEVER = HybridRetriever(docs, DB)
        self.graph = build_graph()

    def stream(self, query):
        result = self.graph.invoke({"query": query})

        if result.get("answer"):
            yield result["answer"]
            return

        out = pipe(result["prompt"])[0]["generated_text"]
        yield out

def build_chatbot():
    return PhysicsChatbot()