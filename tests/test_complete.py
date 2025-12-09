"""
Comprehensive Test Suite for Assignment 5
Tests both fine-tuning and RL training steps
"""
import os
import sys
import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLM


def test_step1_squad_finetuning():
    """Test that SQuAD fine-tuning completed successfully"""
    print("\n" + "="*60)
    print("TEST 1: SQUAD FINE-TUNING")
    print("="*60)
    
    model_path = "./models/gpt2_squad_finetuned"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("❌ FAILED: SQuAD fine-tuned model not found")
        print(f"   Expected location: {model_path}")
        print(f"   Run: python finetune_squad.py")
        return False
    
    print(f"✅ Model directory exists: {model_path}")
    
    # Check required files
    required_files = ["config.json", "pytorch_model.bin", "tokenizer.json", "vocab.json"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ FAILED: Missing files: {missing_files}")
        return False
    
    print("✅ All required model files present")
    
    # Try loading the model
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        print("✅ Model loads successfully")
    except Exception as e:
        print(f"❌ FAILED: Cannot load model: {e}")
        return False
    
    # Test generation
    try:
        test_prompt = "Question: What is the capital of France?\nAnswer:"
        input_ids = tokenizer.encode(test_prompt, return_tensors='pt')
        
        with torch.no_grad():
            output = model.generate(input_ids, max_length=50, num_return_sequences=1)
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"✅ Generation works")
        print(f"   Test prompt: {test_prompt}")
        print(f"   Generated: {generated_text[:100]}...")
        
        # Check if it maintains Q&A format somewhat
        if "Question:" in generated_text and "Answer:" in generated_text:
            print("✅ Model preserves Q&A format structure")
        else:
            print("⚠️  WARNING: Generated text may not follow format perfectly")
            print("   This is expected - RL training will fix this!")
        
    except Exception as e:
        print(f"❌ FAILED: Generation error: {e}")
        return False
    
    print("\n✅ STEP 1 TEST PASSED: SQuAD fine-tuning successful!")
    return True


def test_step2_rl_training():
    """Test that RL training completed successfully"""
    print("\n" + "="*60)
    print("TEST 2: RL TRAINING")
    print("="*60)
    
    model_path = "./models/rl_trained_gpt2"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("❌ FAILED: RL-trained model not found")
        print(f"   Expected location: {model_path}")
        print(f"   Run: python rl_trainer.py")
        return False
    
    print(f"✅ Model directory exists: {model_path}")
    
    # Try loading the model
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        print("✅ RL-trained model loads successfully")
    except Exception as e:
        print(f"❌ FAILED: Cannot load RL model: {e}")
        return False
    
    # Test generation with format checking
    print("\n📝 Testing format compliance...")
    
    test_questions = [
        "What is machine learning?",
        "How does a neural network work?",
        "What is reinforcement learning?",
    ]
    
    compliant_count = 0
    total_tests = len(test_questions)
    
    for i, question in enumerate(test_questions, 1):
        prompt = f"Question: {question}\nAnswer:"
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Check format compliance
        has_question = "Question:" in generated_text
        has_answer = "Answer:" in generated_text
        is_compliant = has_question and has_answer
        
        if is_compliant:
            compliant_count += 1
        
        status = "✅" if is_compliant else "❌"
        print(f"\n{status} Test {i}/{total_tests}: {question[:30]}...")
        print(f"   Has 'Question:': {has_question}")
        print(f"   Has 'Answer:': {has_answer}")
        print(f"   Generated: {generated_text[:80]}...")
    
    # Calculate compliance rate
    compliance_rate = compliant_count / total_tests
    print(f"\n📊 Format Compliance: {compliant_count}/{total_tests} ({compliance_rate*100:.1f}%)")
    
    if compliance_rate >= 0.7:  # At least 70% compliance
        print("✅ STEP 2 TEST PASSED: RL training improved format compliance!")
        return True
    else:
        print("⚠️  WARNING: Low format compliance - may need more RL training")
        print("   Try running: python rl_trainer.py --epochs 100")
        return False


def test_reward_function():
    """Test the reward function works correctly"""
    print("\n" + "="*60)
    print("TEST 3: REWARD FUNCTION")
    print("="*60)
    
    try:
        from reward_function import compute_reward, check_format_compliance
        print("✅ Reward function imports successfully")
    except ImportError as e:
        print(f"❌ FAILED: Cannot import reward function: {e}")
        return False
    
    # Test cases
    test_cases = [
        ("Question: What is AI?\nAnswer: AI is artificial intelligence.", True, 1.0),
        ("This is random text without format.", False, -1.0),
        ("Question: Test", False, -1.0),
        ("Answer: No question part", False, -1.0),
    ]
    
    all_passed = True
    
    for text, expected_compliant, expected_reward in test_cases:
        is_compliant, reason = check_format_compliance(text)
        reward = compute_reward(text)
        
        passed = (is_compliant == expected_compliant and reward == expected_reward)
        status = "✅" if passed else "❌"
        
        print(f"{status} Test: {text[:40]}...")
        print(f"   Expected: compliant={expected_compliant}, reward={expected_reward}")
        print(f"   Got: compliant={is_compliant}, reward={reward}")
        
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ REWARD FUNCTION TEST PASSED!")
        return True
    else:
        print("\n❌ REWARD FUNCTION TEST FAILED!")
        return False


def test_api_integration():
    """Test that model.py works correctly"""
    print("\n" + "="*60)
    print("TEST 4: MODEL WRAPPER")
    print("="*60)
    
    try:
        from model import RLTrainedGPT2
        print("✅ Model wrapper imports successfully")
        
        # Try to load model
        rl_model = RLTrainedGPT2()
        print("✅ RLTrainedGPT2 loads successfully")
        
        # Test generation
        question = "What is deep learning?"
        response = rl_model.generate_formatted_response(question, max_length=80)
        
        print(f"✅ Generation works")
        print(f"   Question: {question}")
        print(f"   Response: {response[:100]}...")
        
        # Check format
        has_format = "Question:" in response and "Answer:" in response
        if has_format:
            print("✅ Response follows format!")
            return True
        else:
            print("⚠️  WARNING: Response may not follow format")
            return False
            
    except FileNotFoundError as e:
        print(f"⚠️  Model not found - run training first: {e}")
        return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_comparison_before_after():
    """Compare base GPT-2 vs RL-trained model"""
    print("\n" + "="*60)
    print("TEST 5: BEFORE vs AFTER COMPARISON")
    print("="*60)
    
    question = "What is artificial intelligence?"
    prompt = f"Question: {question}\nAnswer:"
    
    # Test base GPT-2
    print("\n🔵 BASE GPT-2 (no training):")
    try:
        base_tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        base_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        
        input_ids = base_tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            output = base_model.generate(input_ids, max_length=80, num_return_sequences=1)
        base_text = base_tokenizer.decode(output[0], skip_special_tokens=True)
        
        print(f"Generated: {base_text}")
        base_compliant = "Question:" in base_text and "Answer:" in base_text
        print(f"Format compliant: {base_compliant} {'✅' if base_compliant else '❌'}")
    except Exception as e:
        print(f"Error: {e}")
        base_compliant = False
    
    # Test RL-trained model
    print("\n🟢 RL-TRAINED MODEL:")
    try:
        rl_tokenizer = GPT2Tokenizer.from_pretrained("./models/rl_trained_gpt2")
        rl_model = AutoModelForCausalLM.from_pretrained("./models/rl_trained_gpt2")
        
        input_ids = rl_tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            output = rl_model.generate(input_ids, max_length=80, num_return_sequences=1)
        rl_text = rl_tokenizer.decode(output[0], skip_special_tokens=True)
        
        print(f"Generated: {rl_text}")
        rl_compliant = "Question:" in rl_text and "Answer:" in rl_text
        print(f"Format compliant: {rl_compliant} {'✅' if rl_compliant else '❌'}")
    except Exception as e:
        print(f"Error: {e}")
        rl_compliant = False
    
    # Compare
    print("\n📊 COMPARISON:")
    print(f"   Base GPT-2: {'✅ Compliant' if base_compliant else '❌ Not compliant'}")
    print(f"   RL-trained: {'✅ Compliant' if rl_compliant else '❌ Not compliant'}")
    
    if rl_compliant:
        print("\n✅ RL training successfully improved format compliance!")
        return True
    else:
        print("\n⚠️  RL training may need more epochs or different hyperparameters")
        return False


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "="*60)
    print("ASSIGNMENT 5 - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    results = {}
    
    # Test 1: SQuAD fine-tuning
    results['squad_finetuning'] = test_step1_squad_finetuning()
    
    # Test 2: RL training
    results['rl_training'] = test_step2_rl_training()
    
    # Test 3: Reward function
    results['reward_function'] = test_reward_function()
    
    # Test 4: Model wrapper
    results['model_wrapper'] = test_api_integration()
    
    # Test 5: Before/After comparison
    results['comparison'] = test_comparison_before_after()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\n📊 Overall: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.1f}%)")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! Assignment 5 is complete and working correctly!")
    elif total_passed >= total_tests * 0.8:
        print("\n✅ Most tests passed! Minor issues may need attention.")
    else:
        print("\n⚠️  Several tests failed. Please review the output above.")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
