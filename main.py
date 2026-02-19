from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import torch
import time

# Initialize FastAPI app
app = FastAPI(title="JurisFlow AI Engine")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the lightweight instruction-tuned model
# LaMini-GPT-124M is ~500MB and runs well on CPUs/small servers
MODEL_NAME = "MBZUAI/LaMini-GPT-124M"

print(f"Loading AI model: {MODEL_NAME}...")
try:
    # Use CPU by default for broader compatibility; change to "cuda" if GPU is available
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline("text-generation", model=MODEL_NAME, device=device)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    generator = None

class ContractRequest(BaseModel):
    contract_type: str
    client_name: str
    company_name: str
    jurisdiction: str
    extra_clauses: str = ""

@app.get("/")
async def root():
    return {"status": "JurisFlow AI Engine is running", "model": MODEL_NAME}

@app.post("/generate-contract")
async def generate_contract(request: ContractRequest):
    if not generator:
        raise HTTPException(status_code=503, detail="AI Model not loaded")

    # Construct a detailed prompt for the instruction-tuned model
    prompt = (
        f"Generate a professional {request.contract_type} legal document. "
        f"The agreement is between {request.company_name} (Disclosing Party) and {request.client_name} (Receiving Party). "
        f"The jurisdiction for this contract is {request.jurisdiction}. "
        f"Include the following specific requirements: {request.extra_clauses}. "
        f"The contract should be formal, legally structured, and include standard clauses for {request.contract_type}."
    )

    try:
        start_time = time.time()
        # Generate text with controlled randomness
        results = generator(
            prompt, 
            max_length=800, 
            num_return_sequences=1, 
            temperature=0.7, 
            do_sample=True,
            truncation=True
        )
        
        generated_text = results[0]['generated_text']
        
        # Simple cleanup: Remove the prompt from the output if the model repeats it
        clean_text = generated_text.replace(prompt, "").strip()
        
        return {
            "status": "success",
            "generation_time": f"{round(time.time() - start_time, 2)}s",
            "contract_draft": clean_text,
            "metadata": {
                "type": request.contract_type,
                "jurisdiction": request.jurisdiction,
                "model": MODEL_NAME
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
