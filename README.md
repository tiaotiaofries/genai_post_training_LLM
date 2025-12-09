# Assignment 5: Post-training an LLM using Reinforcement Learning

## Overview

The final model generates responses in this format:
```
Question: [user's question]
Answer: [model's response]
```

## Approach

### Step 1: Fine-tuning on SQuAD Dataset
- **Dataset**: Stanford Question Answering Dataset (SQuAD)
- **Model**: GPT-2 base model (openai-community/gpt2)
- **Training**: Supervised learning on Q&A pairs formatted as "Question: ...\nAnswer: ..."
- **Output**: `./models/gpt2_squad_finetuned/` (~950MB)

### Step 2: RL Post-training
- **Base Model**: SQuAD fine-tuned GPT-2 (from Step 1)
- **Algorithm**: Policy Gradient (REINFORCE)
- **Reward Function**: 
  - `+1` if output follows format: `"Question: ...\nAnswer: ..."`
  - `-1` if output does not follow format
  - Shaped rewards for partial credit
- **Training**: 30-50 epochs with batch size 5
- **Learning Rate**: 1e-5
- **Output**: `./models/rl_trained_gpt2/` (~475MB)

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- PyTorch 2.0+
- transformers 4.30+
- datasets 2.14+
- FastAPI, uvicorn

## Usage

### Step 1: Fine-tune on SQuAD
```bash
python finetune_squad.py --epochs 3 --batch_size 4 --max_samples 10000
```

This will:
- Download SQuAD dataset from HuggingFace
- Fine-tune GPT-2 on Q&A pairs
- Save model to `./models/gpt2_squad_finetuned/`

### Step 2: Train with RL
```bash
python rl_trainer.py
```

This will:
- Load the SQuAD fine-tuned model
- Apply policy gradient RL for format compliance
- Save RL-trained model to `./models/rl_trained_gpt2/`

### Step 3: Run the API Server
```bash
python main.py
# Or with uvicorn:
uvicorn main:app --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000/docs for interactive API documentation.

### Docker Deployment
```bash
docker-compose up --build
```

Or manually:
```bash
docker build -t genai-rl-post-training .
docker run -p 8000:8000 genai-rl-post-training
```

## API Endpoints

### GET `/`
Root endpoint with API information.

### POST `/generate_qa`
Generate a formatted Q&A response.

**Request Body**:
```json
{
  "question": "What is machine learning?",
  "max_length": 100,
  "temperature": 0.7
}
```

**Response**:
```json
{
  "question": "What is machine learning?",
  "formatted_response": "Question: What is machine learning?\nAnswer: Machine learning is...",
  "answer_only": "Machine learning is..."
}
```

### GET `/health`
Health check endpoint.

## Project Structure
```
genai_rl_post_training/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker container build
├── docker-compose.yml         # Docker compose configuration
├── .gitignore                 # Git ignore rules
├── finetune_squad.py          # Step 1: SQuAD fine-tuning
├── reward_function.py         # Format checking reward function
├── rl_trainer.py              # Step 2: RL training
├── model.py                   # RLTrainedGPT2 wrapper class
├── main.py                    # FastAPI application
└── models/
    ├── README.md              # Instructions to regenerate models
    ├── gpt2_squad_finetuned/  # After Step 1 (not in repo, ~950MB)
    └── rl_trained_gpt2/       # After Step 2 (not in repo, ~475MB)
```


## Notes

- **Model files** are excluded from the repository due to size (4.6GB total)
- See `models/README.md` for instructions to regenerate models locally
- Training was tested on Apple Silicon (M-series) with MPS acceleration
- For best results, use GPU/MPS acceleration for faster training
