from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import PhysicsChatbot

app = FastAPI()

bot = PhysicsChatbot()


class Query(BaseModel):
    question: str


@app.post("/chat")
async def chat(q: Query):
    result = await bot.ask_async(q.question)
    return result


@app.get("/")
def root():
    return {"status": "ok"}