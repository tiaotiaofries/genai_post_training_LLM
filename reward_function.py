"""
Reward function for evaluating if GPT-2 output follows the required format.

Format: "Question: [user question]\nAnswer: [model response]"
"""
import re
from typing import Dict, Tuple


def check_format_compliance(text: str, question: str = None) -> Tuple[bool, str]:
    """
    Check if generated text follows the required Q&A format.
    
    Args:
        text: Generated text from the model
        question: Optional original question to verify consistency
        
    Returns:
        Tuple of (is_compliant, reason)
    """
    # Pattern: "Question: <text>\nAnswer: <text>"
    pattern = r"^Question:\s*(.+?)\s*\n+Answer:\s*(.+)$"
    match = re.match(pattern, text.strip(), re.DOTALL | re.IGNORECASE)
    
    if not match:
        # Check what's missing
        if not text.strip().startswith(("Question:", "question:")):
            return False, "Missing 'Question:' prefix"
        elif "\nAnswer:" not in text and "\nanswer:" not in text:
            return False, "Missing 'Answer:' section"
        else:
            return False, "Format malformed"
    
    question_part = match.group(1).strip()
    answer_part = match.group(2).strip()
    
    # Validate non-empty components
    if not question_part:
        return False, "Question part is empty"
    if not answer_part:
        return False, "Answer part is empty"
    
    # If original question provided, check consistency (optional strict mode)
    if question and question.strip().lower() not in question_part.lower():
        return False, "Question doesn't match input"
    
    return True, "Format correct"


def compute_reward(text: str, question: str = None, strict: bool = False) -> float:
    """
    Compute reward based on format compliance.
    
    Args:
        text: Generated text from the model
        question: Optional original question
        strict: If True, requires exact question match
        
    Returns:
        Reward value: +1.0 for correct format, -1.0 for incorrect
    """
    is_compliant, reason = check_format_compliance(text, question if strict else None)
    
    if is_compliant:
        return 1.0
    else:
        return -1.0


def shaped_reward(text: str, question: str = None, partial_credit: bool = True) -> float:
    """
    Shaped reward function with partial credit for partial compliance.
    
    This encourages the model to learn incrementally:
    - Full reward (+1.0) for perfect format
    - Partial reward (+0.5) for having "Question:" 
    - Partial reward (+0.5) for having "Answer:"
    - Negative reward (-1.0) for completely wrong format
    
    Args:
        text: Generated text from the model
        question: Optional original question
        partial_credit: If True, gives partial rewards
        
    Returns:
        Reward value between -1.0 and +1.0
    """
    if not partial_credit:
        return compute_reward(text, question)
    
    is_compliant, reason = check_format_compliance(text, question)
    
    if is_compliant:
        return 1.0
    
    # Give partial credit for partial compliance
    reward = 0.0
    text_lower = text.strip().lower()
    
    if text_lower.startswith("question:"):
        reward += 0.5
    
    if "\nanswer:" in text_lower:
        reward += 0.5
    
    # If no partial compliance at all
    if reward == 0.0:
        return -1.0
    
    return reward


def batch_compute_rewards(
    texts: list[str], 
    questions: list[str] = None,
    shaped: bool = False
) -> list[float]:
    """
    Compute rewards for a batch of generated texts.
    
    Args:
        texts: List of generated texts
        questions: Optional list of original questions
        shaped: If True, use shaped reward function
        
    Returns:
        List of reward values
    """
    if questions is None:
        questions = [None] * len(texts)
    
    reward_fn = shaped_reward if shaped else compute_reward
    
    rewards = []
    for text, question in zip(texts, questions):
        rewards.append(reward_fn(text, question))
    
    return rewards


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_cases = [
        ("Question: What is AI?\nAnswer: AI is artificial intelligence.", True),
        ("Question: How are you?\nAnswer: I'm doing well, thank you!", True),
        ("This is just plain text without format.", False),
        ("Question: What's your name?", False),
        ("Answer: I don't have a name.", False),
        ("Question:\nAnswer: Missing question content.", False),
        ("Question: Valid question\nAnswer:", False),
        ("question: case insensitive?\nanswer: Should work!", True),
    ]
    
    print("=" * 60)
    print("REWARD FUNCTION TESTS")
    print("=" * 60)
    
    for text, expected_compliant in test_cases:
        is_compliant, reason = check_format_compliance(text)
        reward = compute_reward(text)
        shaped_rew = shaped_reward(text)
        
        status = "✅" if is_compliant == expected_compliant else "❌"
        print(f"\n{status} Text: {text[:50]}...")
        print(f"   Compliant: {is_compliant} | Reason: {reason}")
        print(f"   Reward: {reward:.1f} | Shaped: {shaped_rew:.1f}")
    
    print("\n" + "=" * 60)
    print("BATCH REWARD TEST")
    print("=" * 60)
    
    batch_texts = [
        "Question: What is ML?\nAnswer: Machine Learning.",
        "Just random text",
        "Question: Test\nAnswer: Response"
    ]
    
    batch_rewards = batch_compute_rewards(batch_texts, shaped=True)
    print(f"Batch rewards: {batch_rewards}")
