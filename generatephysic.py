import random, json

data = []

for i in range(100):
    V = random.randint(5, 50)
    R = random.randint(1, 20)
    I = V / R

    data.append({
        "id": f"auto_{i}",
        "question": f"Find current if voltage is {V}V and resistance is {R}Ω.",
        "cot": "I = V/R",
        "answer": str(round(I, 2)),
        "unit": "A"
    })

with open("C:/Users/LAPTOP/physics-rag/physic_dataset.json", "w") as f:
    json.dump(data, f, indent=2)