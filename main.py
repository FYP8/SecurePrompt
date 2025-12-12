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
        if result["status"] == "BLOCK":
            print(f"❌ BLOCKED")
            print(f"   Reason: {result['reason']}")
        else:
            print(f"✅ PASSED")
            print(f"   Metrics: {result['metrics']}")
            if "warnings" in result:
                print(f"   ⚠️ Warning: {result['warnings']}")

        print("-" * 40)


if __name__ == "__main__":
    main()