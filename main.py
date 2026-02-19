from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq Settings
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class ContractRequest(BaseModel):
    contract_type: str
    client_name: str
    your_company: str
    jurisdiction: str
    extra_clauses: str

@app.get("/")
def home():
    return {"status": "JurisFlow AI Engine (Groq) is Running"}

@app.post("/generate-contract")
def generate_contract(request: ContractRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is missing in Render settings.")

    prompt = f"Draft a professional {request.contract_type} for {request.client_name} and {request.your_company} in {request.jurisdiction}. Clauses: {request.extra_clauses}"

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a senior lawyer drafting ironclad legal contracts."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return {"error": "Groq API error", "details": response.text}

    result = response.json()
    return {"contract": result['choices'][0]['message']['content']}
    
