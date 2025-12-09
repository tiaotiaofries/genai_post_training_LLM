# Assignment 5: Post-training an LLM using Reinforcement Learning

**Author**: Sze Ning  
**Course**: Applied Gen AI  
**Due**: Sunday 11:59pm  
**Repository**: https://github.com/tiaotiaofries/genai_post_training_LLM

## Overview

This project implements a **two-step approach** to train GPT-2 for formatted Q&A:

1. **Step 1: Supervised Fine-tuning** - Fine-tune GPT-2 on SQuAD dataset to learn Q&A
2. **Step 2: RL Post-training** - Apply Reinforcement Learning to enforce specific format

The final model generates responses in this format:
```
Question: [user's question]
Answer: [model's response]
```

## Approach

### Step 1: Fine-tuning on SQuAD Dataset
- **Dataset**: Stanford Question Answering Dataset (SQuAD)
- **Model**: GPT-2 base model
- **Training**: Supervised learning on Q&A pairs formatted as "Question: ...\nAnswer: ..."
- **Output**: `./models/gpt2_squad_finetuned`

### Step 2: RL Post-training
- **Base Model**: SQuAD fine-tuned GPT-2 (from Step 1)

### RL Training Strategy
- **Algorithm**: Policy Gradient / PPO (Proximal Policy Optimization)
- **Reward Function**: 
  - `+1` if output follows format: `"Question: ...\nAnswer: ..."`
  - `-1` if output does not follow format
- **Training Episodes**: 500-1000 episodes
- **Learning Rate**: 1e-5 (fine-tuning rate for pre-trained LLM)

### Key Components

1. **`finetune_squad.py`**: Supervised fine-tuning on SQuAD dataset
2. **`reward_function.py`**: Evaluates if generated text follows required format
3. **`rl_trainer.py`**: Implements policy gradient training loop
4. **`model.py`**: Wraps GPT-2 with RL-trained policy head
5. **`main.py`**: FastAPI server for inference
6. **`Dockerfile`**: Containerization for deployment

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Fine-tune on SQuAD (First Time Only)
```bash
python finetune_squad.py --epochs 3 --batch_size 4 --max_samples 10000
```

This will:
- Download SQuAD dataset from HuggingFace
- Fine-tune GPT-2 on Q&A pairs
- Save model to `./models/gpt2_squad_finetuned`
- Takes ~30-60 minutes on CPU

### Step 2: Train with RL
```bash
python rl_trainer.py --episodes 1000 --batch_size 30 --lr 1e-5
```

This will:
- Load the SQuAD fine-tuned model
- Apply policy gradient RL
- Save RL-trained model to `./models/rl_trained_gpt2`
- Takes ~20-40 minutes on CPU

### Step 3: Run the API Server
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
├── finetune_squad.py      # Step 1: SQuAD fine-tuning
├── reward_function.py     # Format checking reward
├── rl_trainer.py          # Step 2: RL training
├── model.py               # RL-trained GPT-2 wrapper
├── main.py                # FastAPI application
├── models/                # Saved checkpoints
│   ├── gpt2_squad_finetuned/    # After Step 1
│   └── rl_trained_gpt2/         # After Step 2
└── tests/                 # Unit tests
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
