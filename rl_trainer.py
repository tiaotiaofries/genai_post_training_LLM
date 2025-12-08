"""
RL Trainer for GPT-2 Post-Training
Uses Policy Gradient to train GPT-2 to generate formatted Q&A responses
"""
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, GPT2Tokenizer
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Optional
import os
from reward_function import compute_reward, shaped_reward


class RLTrainer:
    """
    Reinforcement Learning trainer for GPT-2 using Policy Gradient
    
    Similar to Module 11's word chain RL, but applied to full GPT-2 model
    """
    
    def __init__(
        self,
        model_name: str = "openai-community/gpt2",
        base_model_path: Optional[str] = None,
        learning_rate: float = 1e-5,
        device: str = None
    ):
        """
        Initialize RL trainer
        
        Args:
            model_name: HuggingFace model name or path
            base_model_path: Optional path to fine-tuned base model
            learning_rate: Learning rate for RL training
            device: Device to use (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Initializing RL Trainer on {self.device}")
        
        # Load model and tokenizer
        if base_model_path and os.path.exists(base_model_path):
            print(f"📂 Loading base model from {base_model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(base_model_path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_path)
        else:
            print(f"📂 Loading model: {model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        self.model.to(self.device)
        self.model.train()
        
        # Optimizer for policy gradient
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        self.episode_rewards = []
        self.episode_losses = []
        
        print("✅ RL Trainer initialized successfully!")
    
    def generate_episode(
        self,
        question: str,
        max_length: int = 100,
        temperature: float = 0.8
    ) -> Tuple[str, List[int], List[torch.Tensor]]:
        """
        Generate one episode (similar to Module 11's word chain generation)
        
        Args:
            question: Input question
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            (generated_text, token_ids, log_probs)
        """
        # Format the prompt to encourage the desired format
        prompt = f"Question: {question}\nAnswer:"
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Storage for log probabilities (for policy gradient)
        log_probs = []
        token_ids = input_ids[0].tolist()
        
        # Generate tokens one by one (to track log_probs)
        with torch.no_grad():
            for _ in range(max_length):
                # Get model output
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits[:, -1, :]  # Last token logits
                
                # Apply temperature
                logits = logits / temperature
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Store log probability for this action
                log_prob = torch.log(probs[0, next_token.item()])
                log_probs.append(log_prob)
                
                # Append to sequence
                token_ids.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop at EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode generated text
        generated_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        
        return generated_text, token_ids, log_probs
    
    def compute_policy_loss(
        self,
        log_probs: List[torch.Tensor],
        reward: float
    ) -> torch.Tensor:
        """
        Compute policy gradient loss (REINFORCE algorithm)
        
        Similar to Module 10's policy gradient:
        Loss = -sum(log_prob * reward)
        
        Args:
            log_probs: Log probabilities of actions taken
            reward: Total reward for this episode
            
        Returns:
            Policy gradient loss
        """
        # Stack all log probs
        if len(log_probs) == 0:
            return torch.tensor(0.0, device=self.device)
        
        log_probs_tensor = torch.stack(log_probs)
        
        # Policy gradient: maximize reward = minimize negative reward
        # Loss = -reward * sum(log_probs)
        loss = -reward * log_probs_tensor.sum()
        
        return loss
    
    def train_one_epoch(
        self,
        questions: List[str],
        batch_size: int = 10,
        shaped: bool = True,
        max_length: int = 80
    ) -> Tuple[float, float]:
        """
        Train for one epoch (similar to Module 11's training loop)
        
        Args:
            questions: List of training questions
            batch_size: Number of episodes per batch
            shaped: Use shaped reward (partial credit)
            max_length: Max tokens per generation
            
        Returns:
            (average_reward, average_loss)
        """
        total_rewards = []
        total_losses = []
        
        # Process in batches
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            batch_loss = 0.0
            batch_rewards = []
            
            # Generate episodes for batch
            for question in batch_questions:
                # Generate response
                generated_text, token_ids, log_probs = self.generate_episode(
                    question, max_length=max_length
                )
                
                # Compute reward
                reward_fn = shaped_reward if shaped else compute_reward
                reward = reward_fn(generated_text, question)
                
                # Compute loss for this episode
                loss = self.compute_policy_loss(log_probs, reward)
                batch_loss += loss
                
                batch_rewards.append(reward)
            
            # Average loss over batch
            batch_loss = batch_loss / len(batch_questions)
            
            # Backpropagation
            self.optimizer.zero_grad()
            batch_loss.backward()
            
            # Gradient clipping (important for stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_rewards.extend(batch_rewards)
            total_losses.append(batch_loss.item())
        
        avg_reward = np.mean(total_rewards)
        avg_loss = np.mean(total_losses)
        
        return avg_reward, avg_loss
    
    def train(
        self,
        questions: List[str],
        num_epochs: int = 100,
        batch_size: int = 10,
        shaped: bool = True,
        max_length: int = 80,
        save_path: str = "./models/rl_trained_gpt2"
    ):
        """
        Full training loop
        
        Args:
            questions: Training questions
            num_epochs: Number of training epochs
            batch_size: Batch size
            shaped: Use shaped rewards
            max_length: Max generation length
            save_path: Path to save trained model
        """
        print(f"\n{'='*60}")
        print(f"🎯 Starting RL Training")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Questions: {len(questions)}")
        print(f"Shaped rewards: {shaped}")
        print(f"{'='*60}\n")
        
        best_reward = -float('inf')
        
        for epoch in tqdm(range(num_epochs), desc="Training"):
            avg_reward, avg_loss = self.train_one_epoch(
                questions=questions,
                batch_size=batch_size,
                shaped=shaped,
                max_length=max_length
            )
            
            self.episode_rewards.append(avg_reward)
            self.episode_losses.append(avg_loss)
            
            # Print progress every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Reward: {avg_reward:6.3f} | Loss: {avg_loss:8.3f}")
            
            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                self.save_model(save_path)
                print(f"💾 New best reward: {best_reward:.3f} - Model saved!")
        
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"Best reward: {best_reward:.3f}")
        print(f"{'='*60}\n")
        
        # Final save
        self.save_model(save_path)
    
    def save_model(self, save_path: str):
        """Save the trained model"""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def test_generation(self, question: str, num_samples: int = 3):
        """Test the model with a question"""
        print(f"\n{'='*60}")
        print(f"Testing with: {question}")
        print(f"{'='*60}")
        
        for i in range(num_samples):
            generated_text, _, _ = self.generate_episode(question, max_length=80)
            reward = compute_reward(generated_text, question)
            print(f"\nSample {i+1}:")
            print(f"Text: {generated_text}")
            print(f"Reward: {reward}")


# Training script
if __name__ == "__main__":
    # Sample training questions (you can expand this)
    training_questions = [
        "What is machine learning?",
        "How does reinforcement learning work?",
        "What are neural networks?",
        "Explain gradient descent",
        "What is the difference between AI and ML?",
        "How do transformers work?",
        "What is deep learning?",
        "Explain backpropagation",
        "What is natural language processing?",
        "How does GPT work?",
        "What is supervised learning?",
        "Explain overfitting",
        "What are embeddings?",
        "How does attention mechanism work?",
        "What is transfer learning?",
    ]
    
    # Initialize trainer
    trainer = RLTrainer(
        model_name="openai-community/gpt2",
        base_model_path=None,  # Set to fine-tuned model path if available
        learning_rate=1e-5
    )
    
    # Test before training
    print("\n🔍 BEFORE TRAINING:")
    trainer.test_generation("What is AI?", num_samples=2)
    
    # Train
    trainer.train(
        questions=training_questions,
        num_epochs=50,
        batch_size=5,
        shaped=True,
        max_length=80,
        save_path="./models/rl_trained_gpt2"
    )
    
    # Test after training
    print("\n🔍 AFTER TRAINING:")
    trainer.test_generation("What is AI?", num_samples=2)
