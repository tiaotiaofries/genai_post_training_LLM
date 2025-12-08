"""
RL-Trained GPT2 Model for Formatted Q&A Generation
Similar to GPT2QAModel but uses RL-trained weights
"""
import torch
from transformers import AutoModelForCausalLM, GPT2Tokenizer
import os


class RLTrainedGPT2:
    """RL-trained GPT2 model for formatted Q&A generation"""
    
    def __init__(self, model_path="./models/rl_trained_gpt2"):
        """
        Initialize the RL-trained GPT2 model
        
        Args:
            model_path: Path to the RL-trained model directory
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading RL-trained GPT2 model from {model_path}...")
        print(f"Using device: {self.device}")
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"RL-trained model not found at {model_path}. "
                "Please run the RL training script first to create the model."
            )
        
        # Load tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ RL-trained GPT2 model loaded successfully!")
        
    def generate_formatted_response(
        self,
        question: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a formatted Q&A response
        
        Args:
            question: User's question
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text in format "Question: ...\nAnswer: ..."
        """
        # Format the prompt to match training format
        prompt = f"Question: {question}\nAnswer:"
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate text
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length + len(input_ids[0]),  # Add prompt length
                num_return_sequences=1,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text
    
    def generate_answer_only(
        self,
        question: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate only the answer part (without the question prefix)
        
        Args:
            question: User's question
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Just the answer portion
        """
        full_response = self.generate_formatted_response(
            question, max_length, temperature, top_p
        )
        
        # Extract just the answer part
        if "\nAnswer:" in full_response:
            answer = full_response.split("\nAnswer:")[-1].strip()
            return answer
        else:
            # Fallback if format not followed
            return full_response


# Test the model
if __name__ == "__main__":
    import sys
    
    model_path = "./models/rl_trained_gpt2"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Run rl_trainer.py first to train the model.")
        sys.exit(1)
    
    # Load model
    model = RLTrainedGPT2(model_path=model_path)
    
    # Test questions
    test_questions = [
        "What is machine learning?",
        "How does reinforcement learning work?",
        "What are neural networks?",
    ]
    
    print("\n" + "="*60)
    print("TESTING RL-TRAINED MODEL")
    print("="*60)
    
    for question in test_questions:
        print(f"\n📝 Question: {question}")
        response = model.generate_formatted_response(question, max_length=80)
        print(f"🤖 Response:\n{response}")
        print("-" * 60)
