import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# MODEL_DIR = os.getenv("MODEL_PATH", "/model")   # populated by kitops-init
MODEL_DIR = os.getenv("MODEL_PATH", "fine-tuned") #local
tok   = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
gen   = pipeline("text2text-generation", model=model, tokenizer=tok)

app = FastAPI()
class Prompt(BaseModel): text: str

@app.post("/generate")
def generate(p: Prompt):
    return {"response": gen(p.text, max_length=128)[0]["generated_text"]}

@app.get("/health")
def health(): return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)


 