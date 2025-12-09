# Models Directory

This directory contains the trained models for the RL post-training assignment.

## Models

Due to GitHub file size limitations, the trained models are not included in this repository. They need to be generated locally.

### Required Models

1. **gpt2_squad_finetuned/** - GPT-2 fine-tuned on SQuAD dataset (Step 1)
2. **rl_trained_gpt2/** - RL post-trained model (Step 2)

## How to Generate Models

### Step 1: Fine-tune on SQuAD Dataset

```bash
python finetune_squad.py --epochs 3 --batch_size 4 --max_samples 10000
```

This will create `models/gpt2_squad_finetuned/` (~950MB)

**Estimated time:** 1-2 hours on CPU, 20-30 minutes on GPU/MPS

### Step 2: Apply RL Post-Training

```bash
python rl_trainer.py
```

This will create `models/rl_trained_gpt2/` (~475MB)

**Estimated time:** 15-20 minutes on MPS, longer on CPU

## Model Sizes

- `gpt2_squad_finetuned/`: ~950MB (GPT-2 + fine-tuning checkpoints)
- `rl_trained_gpt2/`: ~475MB (RL post-trained model)
- **Total:** ~1.4GB

## Note

The models are automatically saved during training. Make sure you have sufficient disk space before running the training scripts.
