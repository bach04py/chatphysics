import json
import re
import time
from tqdm import tqdm
from chatbot import build_chatbot

# =========================
# CONFIG
# =========================
LEADERBOARD_FILE = "leaderboard.json"

bot = build_chatbot()

# =========================
# HELPERS
# =========================

def normalize(text):
    return text.lower()

def extract_answer(text):
    match = re.search(r"\b([A-D])\b", text.upper())
    if match:
        return match.group(1)

    if "yes" in text.lower():
        return "Yes"
    if "no" in text.lower():
        return "No"

    num = re.search(r"[-+]?\d*\.?\d+", text)
    if num:
        return num.group(0)

    return None

# =========================
# XAI SCORING
# =========================

def keyword_score(pred, gt):
    pred = normalize(pred)
    gt = normalize(gt)

    gt_words = set(gt.split())
    pred_words = set(pred.split())

    overlap = gt_words & pred_words

    return len(overlap) / max(len(gt_words), 1)


def reasoning_score(pred, gt_explanation):
    scores = []

    for gt in gt_explanation:
        scores.append(keyword_score(pred, gt))

    return sum(scores) / len(scores)


# =========================
# LOGIC EVAL
# =========================

def eval_logic(data):

    correct = 0
    total = 0
    reasoning_scores = []

    for item in tqdm(data):

        context = "\n".join(item["premises-NL"])

        for i, q in enumerate(item["questions"]):

            prompt = f"""
Premises:
{context}

Question:
{q}

Answer with reasoning:
"""

            response = "".join(bot.stream_answer(prompt))

            pred = extract_answer(response)
            gt = item["answers"][i]

            if str(pred) == str(gt):
                correct += 1

            # XAI scoring
            r_score = reasoning_score(response, item["explanation"])
            reasoning_scores.append(r_score)

            total += 1

    acc = correct / total
    avg_reasoning = sum(reasoning_scores) / len(reasoning_scores)

    return acc, avg_reasoning


# =========================
# PHYSICS EVAL
# =========================

def eval_physics(data):

    correct = 0
    total = 0
    reasoning_scores = []

    for item in tqdm(data):

        response = "".join(bot.stream_answer(item["question"]))

        pred = extract_answer(response)
        gt = item["answer"]

        try:
            if abs(float(pred) - float(gt)) < 1e-2:
                correct += 1
        except:
            pass

        r_score = keyword_score(response, item["cot"])
        reasoning_scores.append(r_score)

        total += 1

    acc = correct / total
    avg_reasoning = sum(reasoning_scores) / len(reasoning_scores)

    return acc, avg_reasoning


# =========================
# LEADERBOARD
# =========================

def save_result(result):

    try:
        with open(LEADERBOARD_FILE, "r") as f:
            board = json.load(f)
    except:
        board = []

    board.append(result)

    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(board, f, indent=2)


def show_leaderboard():

    try:
        with open(LEADERBOARD_FILE) as f:
            board = json.load(f)
    except:
        print("No leaderboard yet")
        return

    board = sorted(board, key=lambda x: x["score"], reverse=True)

    print("\n🏆 LEADERBOARD")
    print("-" * 40)

    for i, r in enumerate(board[:5]):
        print(f"{i+1}. {r['name']} | Score: {r['score']:.3f}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    with open("logic_dataset.json") as f:
        logic_data = json.load(f)

    with open("physics_dataset.json") as f:
        physics_data = json.load(f)

    print("🚀 Running evaluation...")

    start = time.time()

    logic_acc, logic_reason = eval_logic(logic_data)
    phys_acc, phys_reason = eval_physics(physics_data)

    total_score = (
        0.4 * logic_acc +
        0.4 * phys_acc +
        0.2 * (logic_reason + phys_reason) / 2
    )

    result = {
        "name": "PhysicsBot-v1",
        "logic_acc": logic_acc,
        "physics_acc": phys_acc,
        "logic_reason": logic_reason,
        "physics_reason": phys_reason,
        "score": total_score,
        "time": time.time() - start
    }

    print("\n📊 RESULTS")
    print(result)

    save_result(result)
    show_leaderboard()