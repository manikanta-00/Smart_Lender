import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smart_lender import train, app  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_store", "model.pkl")


def main():
    force_retrain = "--retrain" in sys.argv

    if force_retrain or not os.path.exists(MODEL_PATH):
        print("=" * 70)
        print(" SMART LENDER - First-time setup: generating data & training models")
        print("=" * 70)
        train.run_pipeline(force_regenerate_data=force_retrain)
    else:
        print("[run] Existing trained model found -- skipping training.")
        print("[run] (Run `python run.py --retrain` to regenerate data & retrain.)")

    app.run(host="127.0.0.1", port=5000)


if __name__ == "__main__":
    main()
