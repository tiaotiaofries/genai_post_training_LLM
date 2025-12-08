# Quick Start Guide - Assignment 5

## 🚀 Step-by-Step Execution

### Step 1: Train the RL Model (FIRST TIME ONLY)
```bash
cd /Users/szening/genai_rl_post_training
python rl_trainer.py
```
**Expected Output:**
- Training progress bars
- Epoch updates with rewards
- Model saved to `./models/rl_trained_gpt2/`
- Training time: ~10-20 minutes on CPU

### Step 2: Test the Standalone API
```bash
python main.py
```
Then visit: http://localhost:8000/docs

### Step 3: Run with Docker
```bash
docker-compose up --build
```

### Step 4: Test the API
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

- [ ] Model trains without errors
- [ ] API starts successfully
- [ ] `/generate_qa` endpoint returns formatted responses
- [ ] Docker container builds and runs
- [ ] Format follows: "Question: ...\nAnswer: ..."

## 🐛 Troubleshooting

**Model not found:**
```bash
# Make sure you ran training first
python rl_trainer.py
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
