"""
Fine-tune GPT-2 on SQuAD Dataset
Assignment 5 - Step 1: Supervised Fine-tuning before RL
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from tqdm import tqdm
import os


def prepare_squad_data(examples, tokenizer, max_length=512):
    """
    Prepare SQuAD data for GPT-2 fine-tuning
    
    Format: "Question: [question]\nAnswer: [answer]"
    This matches the format we want RL to learn!
    """
    formatted_texts = []
    
    for question, answers in zip(examples['question'], examples['answers']):
        # SQuAD answers is a dict with 'text' list
        answer_text = answers['text'][0] if answers['text'] else "No answer"
        
        # Format as Q&A (this is what we want the model to learn)
        formatted = f"Question: {question}\nAnswer: {answer_text}"
        formatted_texts.append(formatted)
    
    # Tokenize
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized['labels'] = tokenized['input_ids'].clone()
    
    return tokenized


def fine_tune_on_squad(
    model_name: str = "openai-community/gpt2",
    output_dir: str = "./models/gpt2_squad_finetuned",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_samples: int = 10000  # Use subset for faster training
):
    """
    Fine-tune GPT-2 on SQuAD dataset
    
    Args:
        model_name: Base model to fine-tune
        output_dir: Where to save the fine-tuned model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_samples: Max training samples (use subset for speed)
    """
    print("="*60)
    print("FINE-TUNING GPT-2 ON SQUAD DATASET")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"\n📂 Loading model: {model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    print("✅ Model loaded")
    
    # Load SQuAD dataset
    print("\n📊 Loading SQuAD dataset...")
    dataset = load_dataset("rajpurkar/squad")
    
    print(f"Total training examples: {len(dataset['train'])}")
    print(f"Using subset: {max_samples} examples")
    
    # Use subset for faster training
    train_dataset = dataset['train'].select(range(min(max_samples, len(dataset['train']))))
    eval_dataset = dataset['validation'].select(range(min(1000, len(dataset['validation']))))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    # Show example
    print("\n📝 Example from dataset:")
    example = train_dataset[0]
    print(f"Context: {example['context'][:100]}...")
    print(f"Question: {example['question']}")
    print(f"Answer: {example['answers']['text'][0]}")
    
    formatted_example = f"Question: {example['question']}\nAnswer: {example['answers']['text'][0]}"
    print(f"\n📋 Formatted for training:\n{formatted_example}")
    
    # Prepare datasets
    print("\n🔄 Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: prepare_squad_data(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    eval_dataset = eval_dataset.map(
        lambda x: prepare_squad_data(x, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    print("✅ Tokenization complete")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=100,
        eval_steps=500,
        save_steps=500,
        eval_strategy="steps",  # Changed from evaluation_strategy
        save_total_limit=2,
        load_best_model_at_end=True,
        warmup_steps=100,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to="none"  # Disable wandb/tensorboard
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n🚀 Starting fine-tuning...")
    print(f"This will take approximately {num_epochs * len(train_dataset) // batch_size // 60} minutes")
    
    trainer.train()
    
    # Save final model
    print(f"\n💾 Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "="*60)
    print("✅ FINE-TUNING COMPLETE!")
    print("="*60)
    print(f"Model saved to: {output_dir}")
    print("\nNext step: Run RL training on this fine-tuned model")
    print("  python rl_trainer.py")
    
    return model, tokenizer


def test_finetuned_model(model_path: str = "./models/gpt2_squad_finetuned"):
    """Test the fine-tuned model"""
    print("\n" + "="*60)
    print("TESTING FINE-TUNED MODEL")
    print("="*60)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Run fine-tuning first!")
        return
    
    # Load model
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Test questions
    test_questions = [
        "What is machine learning?",
        "How does a neural network work?",
        "What is the capital of France?",
    ]
    
    print("\n📝 Generating responses...\n")
    
    for question in test_questions:
        prompt = f"Question: {question}\nAnswer:"
        
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}")
        print("-" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on SQuAD")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_samples", type=int, default=10000, help="Max training samples")
    parser.add_argument("--test_only", action="store_true", help="Only test existing model")
    
    args = parser.parse_args()
    
    if args.test_only:
        test_finetuned_model()
    else:
        # Fine-tune
        model, tokenizer = fine_tune_on_squad(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            max_samples=args.max_samples
        )
        
        # Test
        test_finetuned_model()
