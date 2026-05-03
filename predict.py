"""
predict.py — Run inference with the trained spam classifier.

Usage:
    python predict.py
    python predict.py --message "Congratulations! You won a prize!"
"""

import pickle
import argparse


def load_model(path="models/spam_classifier.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(model, messages):
    preds = model.predict(messages)
    probs = model.predict_proba(messages)
    results = []
    for msg, pred, prob in zip(messages, preds, probs):
        label = "SPAM" if pred == 1 else "HAM"
        confidence = prob[pred]
        results.append({"message": msg, "label": label, "confidence": f"{confidence:.2%}"})
    return results


def main():
    parser = argparse.ArgumentParser(description="Email Spam Classifier")
    parser.add_argument("--message", type=str, help="Message to classify")
    args = parser.parse_args()

    model = load_model()

    if args.message:
        test_messages = [args.message]
    else:
        # Demo messages
        test_messages = [
            "Hey, are you coming to the meeting today?",
            "WINNER!! You have been selected for a £900 prize reward! Call 09061701461 now!",
            "Can you pick up some milk on the way home?",
            "FREE entry: Win a brand new iPhone! Text WIN to 80085 now. Premium rate applies.",
            "The project deadline is next Friday, don't forget.",
            "Congratulations ur awarded either £500 of CD gift vouchers or £125 cash gift guaranteed!",
        ]

    print("\n" + "=" * 65)
    print("  SPAM CLASSIFIER — PREDICTIONS")
    print("=" * 65)

    results = predict(model, test_messages)
    for r in results:
        icon = "🚫" if r["label"] == "SPAM" else "✅"
        print(f"\n{icon} [{r['label']}] (confidence: {r['confidence']})")
        print(f"   {r['message'][:80]}{'...' if len(r['message']) > 80 else ''}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
