"""
Quick validation script - checks if models exist without running generation
Fast check before starting training or deployment
"""
import os


def quick_check():
    """Quick check of project status"""
    print("="*60)
    print("ASSIGNMENT 5 - QUICK STATUS CHECK")
    print("="*60)
    
    checks = {
        "SQuAD fine-tuned model": "./models/gpt2_squad_finetuned",
        "RL-trained model": "./models/rl_trained_gpt2",
        "Reward function": "reward_function.py",
        "RL trainer": "rl_trainer.py",
        "Fine-tuning script": "finetune_squad.py",
        "Model wrapper": "model.py",
        "API server": "main.py",
        "Dockerfile": "Dockerfile",
        "Tests": "tests/test_api.py",
    }
    
    print("\n📋 Checking files and models...")
    
    status = {}
    for name, path in checks.items():
        exists = os.path.exists(path)
        status[name] = exists
        icon = "✅" if exists else "❌"
        print(f"{icon} {name}: {path}")
    
    # Next steps
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    
    if not status["SQuAD fine-tuned model"]:
        print("\n🔵 STEP 1: Fine-tune on SQuAD")
        print("   python finetune_squad.py --epochs 3")
        print("   ⏱  Time: ~30-60 minutes")
    elif not status["RL-trained model"]:
        print("\n🟢 STEP 2: RL Training")
        print("   python rl_trainer.py")
        print("   ⏱  Time: ~20-40 minutes")
    else:
        print("\n🎉 Both models trained!")
        print("\n✅ Run comprehensive tests:")
        print("   python tests/test_complete.py")
        print("\n✅ Start API server:")
        print("   python main.py")
        print("\n✅ Deploy with Docker:")
        print("   docker-compose up --build")
    
    # Statistics
    total = len(status)
    completed = sum(status.values())
    print(f"\n📊 Progress: {completed}/{total} ({completed/total*100:.0f}%)")
    
    return all(status.values())


if __name__ == "__main__":
    all_ready = quick_check()
    
    if all_ready:
        print("\n🎓 Assignment 5 is ready for submission!")
    else:
        print("\n⚠️  Complete the steps above before submission.")
