from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM

# Initialize the FastAPI app
app = FastAPI()

# Load the LLM model
llm = OllamaLLM(model="llama3.1")


# Define the request body model
class QueryRequest(BaseModel):
    query: str


# Define the response model
class QueryResponse(BaseModel):
    response: str


@app.post("/generate-response", response_model=QueryResponse)
async def generate_response(request: QueryRequest):
    try:
        # Use the LLM to generate a response
        response = llm.invoke(request.query)
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
