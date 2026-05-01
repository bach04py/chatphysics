import os
import asyncio
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS

# =========================
# ENV
# =========================
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "llama3")
DATA_PATH = os.getenv("DATA_PATH", "./data/physics.pdf")
VECTOR_DB_PATH = "./faiss_index"

# =========================
# LLM
# =========================
def build_llm(num_predict=100, temperature=0.2):
    return Ollama(
        model=MODEL_NAME,
        temperature=temperature,
        num_predict=num_predict
    )

async def llm_call(llm, prompt):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, llm.invoke, prompt)

# =========================
# DATA
# =========================
def load_pdf():
    return PyPDFLoader(DATA_PATH).load()

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )
    return splitter.split_documents(docs)

def build_db(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTOR_DB_PATH)
    return db

# =========================
# ROUTER (multi-agent logic)
# =========================
def classify(query):
    q = query.lower()

    if any(k in q for k in ["calculate", "tính", "find"]):
        return "reasoning"

    if any(k in q for k in ["relationship", "related", "liên hệ"]):
        return "relation"

    return "simple"

# =========================
# MEMORY
# =========================
class Memory:
    def __init__(self):
        self.history = []

    def add(self, user, bot):
        self.history.append((user, bot))
        self.history = self.history[-3:]

    def get_context(self):
        return "\n".join([f"User:{u}\nBot:{b}" for u, b in self.history])

# =========================
# MAIN CHATBOT
# =========================
class PhysicsChatbot:

    def __init__(self):
        docs = split_docs(load_pdf())
        self.db = build_db(docs)
        self.memory = Memory()
        self.cache = {}

    def select_llm(self, route):
        if route == "simple":
            return build_llm(60, 0.1)
        if route == "relation":
            return build_llm(120, 0.2)
        return build_llm(200, 0.3)

    async def ask_async(self, query):

        if query in self.cache:
            return self.cache[query]

        route = classify(query)
        llm = self.select_llm(route)

        docs = self.db.similarity_search(query, k=2)
        context = "\n".join([d.page_content[:200] for d in docs])

        memory_context = self.memory.get_context()

        # =========================
        # SIMPLE
        # =========================
        if route == "simple":
            prompt = f"""
Answer briefly.

Context:
{context}

Conversation:
{memory_context}

Question:
{query}
"""
            answer = await llm_call(llm, prompt)

        # =========================
        # RELATION
        # =========================
        elif route == "relation":
            prompt = f"""
Explain relationship clearly.

Context:
{context}

Question:
{query}
"""
            answer = await llm_call(llm, prompt)

        # =========================
        # REASONING (multi-step)
        # =========================
        else:
            summary = await llm_call(
                build_llm(100, 0.2),
                f"Summarize:\n{context}"
            )

            answer = await llm_call(
                llm,
                f"""
Solve step by step.

Question:
{query}

Context:
{summary}
"""
            )

        result = {
            "answer": answer,
            "route": route,
            "context": context
        }

        self.memory.add(query, answer)
        self.cache[query] = result

        return result

    def ask(self, query):
        return asyncio.run(self.ask_async(query))

# =========================
# FACTORY (fix ImportError)
# =========================
def build_chatbot():
    return PhysicsChatbot()