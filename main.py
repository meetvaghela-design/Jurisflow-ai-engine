from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os

app = FastAPI()

# CORS allow karna taaki Next.js frontend se connect ho sake
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face Settings (Aap apni API Key baad mein Render mein daal sakte hain)
# Model: Meta Llama 3
HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

class ContractRequest(BaseModel):
    contract_type: str
    client_name: str
    your_company: str
    jurisdiction: str
    extra_clauses: str

@app.get("/")
def home():
    return {"status": "JurisFlow AI Engine is Running"}

@app.post("/generate-contract")
def generate_contract(request: ContractRequest):
    prompt = f"""
    Act as a Senior Corporate Lawyer. Draft a highly professional {request.contract_type}.
    - Client Name: {request.client_name}
    - Provider/Company: {request.your_company}
    - Jurisdiction: {request.jurisdiction}
    - Specific Requirements: {request.extra_clauses}
    
    Ensure the document is legally sound, uses formal terminology, and includes standard clauses for {request.jurisdiction} law.
    """

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 1000, "temperature": 0.7}
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="AI Model is busy or Token is missing.")

    result = response.json()
    return {"contract": result[0]['generated_text']}
    
