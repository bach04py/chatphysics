import time
import json
from typing import List, Dict
from chatbot import PhysicsChatbot


# =========================
# TEST DATASET
# =========================
TEST_CASES = [

    {
        "type": "theory",
        "question": "What is Newton's second law?",
        "keywords": ["force", "mass", "acceleration"]
    },
    {
        "type": "graph",
        "question": "How is force related to acceleration?",
        "keywords": ["force", "acceleration"]
    },
    {
        "type": "calculation",
        "question": "A 4 kg object is acted on by a 20 N force. Find acceleration.",
        "keywords": ["5"]
    },
    {
        "type": "theory",
        "question": "Define kinetic energy.",
        "keywords": ["mass", "velocity"]
    },
    {
        "type": "calculation",
        "question": "Calculate kinetic energy of a 2 kg object moving at 3 m/s.",
        "keywords": ["9"]
    },
    {
        "type": "memory",
        "question": "What is force?",
        "keywords": ["mass", "acceleration"]
    },
    {
        "type": "memory",
        "question": "What is its formula?",
        "keywords": ["f", "ma"]
    },
    {
        "type": "multi",
        "question": "Explain step-by-step how energy is conserved in a pendulum.",
        "keywords": ["energy", "kinetic", "potential"]
    }
]


# =========================
# SCORING
# =========================
def keyword_score(answer: str, keywords: List[str]) -> int:
    answer = answer.lower()
    return sum(1 for kw in keywords if kw.lower() in answer)


def normalize(score: int, total: int):
    return round(score / total, 2)


# =========================
# EVALUATION
# =========================
def evaluate(bot: PhysicsChatbot, debug=True, save_log=True):

    results = []

    print("\n=== RUNNING EVALUATION ===\n")

    for i, case in enumerate(TEST_CASES):

        q = case["question"]
        expected = case["keywords"]

        print(f"[{i+1}] {q}")

        # =========================
        # RUN BOT
        # =========================
        start = time.time()
        output = bot.ask(q)
        latency = round(time.time() - start, 2)

        # =========================
        # HANDLE OUTPUT
        # =========================
        if isinstance(output, dict):
            answer = output.get("answer", "")
            q_type = output.get("type", "unknown")
            context = output.get("context", "")
            graph = output.get("graph", "")
            summary = output.get("summary", "")
            reasoning = output.get("reasoning", "")
        else:
            answer = output
            q_type = "unknown"
            context = graph = summary = reasoning = ""

        # =========================
        # SCORING
        # =========================
        score = keyword_score(answer, expected)
        norm = normalize(score, len(expected))

        # =========================
        # DEBUG PRINT
        # =========================
        if debug:
            print("="*70)

            print(f"🧠 TYPE: {q_type}")

            print("\n📥 CONTEXT:")
            print(context[:500])

            print("\n🕸 GRAPH:")
            print(graph[:300])

            print("\n📚 SUMMARY:")
            print(summary[:300])

            print("\n🧮 REASONING:")
            print(reasoning[:300])

            print("\n🤖 ANSWER:")
            print(answer)

            print(f"\n📊 Score: {score}/{len(expected)} | {norm}")
            print(f"⏱ Latency: {latency}s")

            print("="*70, "\n")

        # =========================
        # SAVE RESULT
        # =========================
        result = {
            "question": q,
            "type": case["type"],
            "predicted_type": q_type,
            "score": score,
            "normalized": norm,
            "latency": latency,
            "answer": answer,
            "context": context,
            "graph": graph,
            "summary": summary,
            "reasoning": reasoning
        }

        results.append(result)

    # =========================
    # SAVE LOG
    # =========================
    if save_log:
        with open("evaluation_log.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results


# =========================
# SUMMARY
# =========================
def summarize(results: List[Dict]):

    total = len(results)
    avg_score = sum(r["normalized"] for r in results) / total

    print("\n=== SUMMARY ===")
    print(f"Overall Score: {round(avg_score, 2)}")

    # breakdown by type
    by_type = {}

    for r in results:
        t = r["type"]
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(r["normalized"])

    print("\nBreakdown:")
    for t, vals in by_type.items():
        print(f"{t}: {round(sum(vals)/len(vals), 2)}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    bot = PhysicsChatbot()

    results = evaluate(bot, debug=True, save_log=True)

    summarize(results)