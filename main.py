"""
FastAPI Application for RL-Trained GPT2 Q&A
Assignment 5: Post-training an LLM using Reinforcement Learning
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from model import RLTrainedGPT2
import os

app = FastAPI(
    title="RL Post-Trained GPT2 API",
    version="1.0",
    description="GPT-2 model trained with RL to generate formatted Q&A responses"
)

# Load RL-trained model
try:
    rl_model = RLTrainedGPT2(model_path="./models/rl_trained_gpt2")
    model_available = True
except FileNotFoundError as e:
    print(f"⚠️  RL-trained model not available: {e}")
    print("   Run rl_trainer.py to train the model first.")
    rl_model = None
    model_available = False


class QuestionRequest(BaseModel):
    """Request model for Q&A generation"""
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The question to answer",
        examples=["What is machine learning?"]
    )
    max_length: int = Field(
        100,
        ge=10,
        le=300,
        description="Maximum tokens in the response"
    )
    temperature: float = Field(
        0.7,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (higher = more random)"
    )


class QuestionResponse(BaseModel):
    """Response model for Q&A generation"""
    question: str
    formatted_response: str
    answer_only: str


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "service": "RL Post-Trained GPT2 Q&A API",
        "status": "online",
        "model_available": model_available,
        "endpoints": {
            "/generate_qa": "Generate formatted Q&A response",
            "/health": "Health check",
            "/docs": "API documentation"
        },
        "format": "Question: [question]\\nAnswer: [answer]"
    }


@app.post("/generate_qa", response_model=QuestionResponse)
def generate_qa(request: QuestionRequest):
    """
    Generate a formatted Q&A response using RL-trained GPT2
    
    The model has been trained with reinforcement learning to follow the format:
    Question: [user's question]
    Answer: [model's response]
    
    Args:
        request: Contains question, max_length, and temperature
        
    Returns:
        Formatted Q&A response with both full format and answer-only
    """
    if not model_available:
        raise HTTPException(
            status_code=503,
            detail="RL-trained model not available. Please train the model first using rl_trainer.py"
        )
    
    try:
        # Generate formatted response
        formatted_response = rl_model.generate_formatted_response(
            question=request.question,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        # Generate answer-only version
        answer_only = rl_model.generate_answer_only(
            question=request.question,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        return QuestionResponse(
            question=request.question,
            formatted_response=formatted_response,
            answer_only=answer_only
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_available,
        "device": str(rl_model.device) if model_available else "N/A"
    }


# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
