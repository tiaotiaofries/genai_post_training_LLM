# Quick Start Guide - Assignment 5

## 🚀 Two-Step Process

Assignment 5 requires **two steps**:
1. **Fine-tune GPT-2 on SQuAD** (supervised learning)
2. **Apply RL post-training** (reinforcement learning)

## 📋 Step-by-Step Execution

### Step 1: Fine-tune on SQuAD Dataset (REQUIRED FIRST)
```bash
cd /Users/szening/genai_rl_post_training
python finetune_squad.py --epochs 3 --batch_size 4 --max_samples 10000
```

**What this does:**
- Downloads SQuAD dataset from HuggingFace
- Formats data as "Question: ...\nAnswer: ..."
- Fine-tunes GPT-2 for 3 epochs
- Saves to `./models/gpt2_squad_finetuned/`

**Expected Output:**
- Training progress with loss decreasing
- Validation metrics
- Model saved successfully
- **Time: ~30-60 minutes on CPU, ~10-15 minutes on GPU**

**Test the fine-tuned model:**
```bash
python finetune_squad.py --test_only
```

### Step 2: RL Post-Training
```bash
python rl_trainer.py
```

**What this does:**
- Loads SQuAD fine-tuned model from Step 1
- Applies policy gradient RL
- Enforces format compliance
- Saves to `./models/rl_trained_gpt2/`

**Expected Output:**
- Epoch-by-epoch training progress
- Rewards increasing (from ~0 to ~0.8+)
- Best model automatically saved
- **Time: ~20-40 minutes on CPU**

### Step 3: Test the Standalone API
```bash
python main.py
```
Then visit: http://localhost:8000/docs

### Step 4: Run with Docker
```bash
docker-compose up --build
```

### Step 5: Test the API
```bash
python tests/test_api.py
```

## 📋 What to Submit

1. **GitHub Repository**: All code pushed to `genai_post_training_LLM`
2. **Docker**: Working Dockerfile and docker-compose.yml
3. **API**: Functional `/generate_qa` endpoint
4. **Documentation**: README.md with usage instructions
5. **Theoretical Answers**: Separate document (30 points)

## 🧪 Example API Request

```bash
curl -X POST "http://localhost:8000/generate_qa" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is reinforcement learning?",
    "max_length": 100,
    "temperature": 0.7
  }'
```

**Expected Response:**
```json
{
  "question": "What is reinforcement learning?",
  "formatted_response": "Question: What is reinforcement learning?\nAnswer: Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment...",
  "answer_only": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment..."
}
```

## ✅ Success Criteria

- [ ] **Step 1**: SQuAD fine-tuning completes successfully
- [ ] **Step 2**: RL training runs without errors
- [ ] API starts successfully
- [ ] `/generate_qa` endpoint returns formatted responses
- [ ] Docker container builds and runs
- [ ] Format follows: "Question: ...\nAnswer: ..."

## 🐛 Troubleshooting

**"Model not found" error in RL training:**
```bash
# Make sure you ran Step 1 first!
python finetune_squad.py
```

**SQuAD dataset download fails:**
```bash
# Check internet connection
# HuggingFace datasets requires internet access
```

**Import errors:**
```bash
# Install dependencies
pip install -r requirements.txt
```

**Docker issues:**
```bash
# Rebuild
docker-compose down
docker-compose up --build
```

## 📊 Training Parameters (Adjustable)

In `rl_trainer.py`, modify:
- `num_epochs`: 50 (increase for better results)
- `batch_size`: 5 (increase if you have GPU)
- `learning_rate`: 1e-5 (lower for stability)
- `shaped`: True (use shaped rewards)

## Next: Push to GitHub
```bash
cd /Users/szening/genai_rl_post_training
git add .
git commit -m "Assignment 5: RL post-training implementation"
git push origin main
```
