import sys
from src.monitors import SecurePromptPipeline


def main():
    print("========================================")
    print("   SecurePrompt - CLI Test Interface    ")
    print("========================================")

    # 1. Initialize Pipeline
    try:
        pipeline = SecurePromptPipeline()
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load pipeline. {e}")
        sys.exit(1)

    print("\n[INFO] System loaded. Type 'exit' to quit.\n")

    # 2. Loop for user input
    while True:
        user_input = input("Enter Prompt to Scan >> ")

        if user_input.lower() in ["exit", "quit"]:
            break

        if not user_input.strip():
            continue

        # 3. Scan
        print("Scanning...")
        result = pipeline.scan_input(user_input)

        # 4. Display Results
        risk = result.get("total_risk", 0.0)
        breakdown = result.get("breakdown", {})

        print("-" * 50)
        if result["status"] == "BLOCK":
            print(f"❌ BLOCKED (Risk Score: {risk:.4f} / Threshold: 0.5)")
            print(f"   Reason: {result['reason']}")
        else:
            print(f"✅ PASSED (Risk Score: {risk:.4f})")

        # Show the "Why" (The Ensemble Voting)
        if breakdown:
            print(f"\n   📊 Model Voting Breakdown:")
            print(f"      • Heuristic (Regex/Keys): {breakdown.get('heuristic_score', 0.0)} (Weight: 0.2)")
            print(f"      • Perplexity (Gibberish): {breakdown.get('perplexity_norm', 0.0)} (Weight: 0.3)")
            print(f"      • BERT AI (Semantic):     {breakdown.get('bert_prob', 0.0):.4f} (Weight: 0.5)")
        print("-" * 50)


if __name__ == "__main__":
    main()