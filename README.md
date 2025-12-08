# Assignment 5: Post-training an LLM using Reinforcement Learning

**Author**: Sze Ning  
**Course**: Applied Gen AI  
**Due**: Sunday 11:59pm  
**Repository**: https://github.com/tiaotiaofries/genai_post_training_LLM

## Overview

This project implements Reinforcement Learning (RL) to post-train GPT-2 to generate responses in a specific format:
```
Question: [user's question]
Answer: [model's response]
```

## Approach

### Base Model
- **Model**: GPT-2 fine-tuned on Nectar QA dataset (from Module 9)
- **Location**: `/Users/szening/sps_genai/models/gpt2_finetuned`

### RL Training Strategy
- **Algorithm**: Policy Gradient / PPO (Proximal Policy Optimization)
- **Reward Function**: 
  - `+1` if output follows format: `"Question: ...\nAnswer: ..."`
  - `-1` if output does not follow format
- **Training Episodes**: 500-1000 episodes
- **Learning Rate**: 1e-5 (fine-tuning rate for pre-trained LLM)

### Key Components

1. **`reward_function.py`**: Evaluates if generated text follows required format
2. **`rl_trainer.py`**: Implements policy gradient training loop
3. **`model.py`**: Wraps GPT-2 with RL-trained policy head
4. **`main.py`**: FastAPI server for inference
5. **`Dockerfile`**: Containerization for deployment

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train the Model
```bash
python rl_trainer.py --episodes 1000 --batch_size 30 --lr 1e-5
```

### Run the API Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
docker build -t genai-rl-post-training .
docker run -p 8000:8000 genai-rl-post-training
```

## API Endpoints

### POST `/generate`
Generate a formatted response to a user question.

**Request Body**:
```json
{
  "question": "What is machine learning?",
  "max_length": 100
}
```

**Response**:
```json
{
  "response": "Question: What is machine learning?\nAnswer: Machine learning is a subset of artificial intelligence..."
}
```

## Project Structure
```
genai_rl_post_training/
├── README.md
├── requirements.txt
├── Dockerfile
├── .gitignore
├── reward_function.py    # Format checking reward
├── rl_trainer.py         # RL training implementation
├── model.py              # RL-trained GPT-2 wrapper
├── main.py               # FastAPI application
├── models/               # Saved RL checkpoints
└── tests/                # Unit tests
```

## RL Concepts Applied

- **States**: GPT-2 hidden states during generation
- **Actions**: Token selections at each timestep
- **Policy**: GPT-2's probability distribution over vocabulary
- **Reward**: Format compliance score (+1/-1)
- **Policy Gradient**: Update policy to maximize expected reward

## Grading Criteria
- ✅ GitHub commit (10 pts)
- ✅ Docker + FastAPI (20 pts)
- ✅ API queries work (20 pts)
- ✅ Code organization (20 pts)
- ✅ Theoretical questions (30 pts)

## References
- Module 10: RL Basics (Grid World)
- Module 11: RL for Language Models (Word Chains)
- TRL Library: https://github.com/huggingface/trl
