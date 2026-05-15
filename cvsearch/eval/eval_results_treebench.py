import json
import argparse
import numpy as np

def eval_treebench(answer_file):
    tags = [
        "Perception/Attributes",
        "Perception/Material",
        "Perception/Physical State",
        "Perception/Object Retrieval",
        "Perception/OCR",
        "Reasoning/Perspective Transform",
        "Reasoning/Ordering",
        "Reasoning/Contact and Occlusion",
        "Reasoning/Spatial Containment",
        "Reasoning/Comparison"
    ]
    data = []
    with open(answer_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                item["prediction"] = item.get("output", "")
                data.append(item)

    results = {}
    total = 0
    correct = 0

    for tag in tags:
        results[tag] = {"correct": 0, "total": 0}

    for item in data:
        category = item.get("category")
        if category not in tags:
            continue

        results[category]["total"] += 1
        total += 1
        pred = str(item["prediction"]).strip().upper()
        ans = str(item["answer"]).strip().upper()
        if pred == ans:
            results[category]["correct"] += 1
            correct += 1

    for tag in tags:
        if results[tag]["total"] == 0:
            acc = 0.0
            print(f"{tag}: 0/0=0.00")
        else:
            acc = results[tag]["correct"] / results[tag]["total"]
            print(f"{tag}: {results[tag]['correct']}/{results[tag]['total']}={round(acc * 100, 2)}")

    if total == 0:
        overall_acc = 0.0
        print("==> Overall: 0/0=0.00")
    else:
        overall_acc = correct / total
        print(f"==> Overall: {correct}/{total}={round(overall_acc * 100, 2)}")

    iou_list = []
    for item in data:
        if "iou" in item and item["iou"] is not None:
            try:
                iou_val = float(item["iou"])
                if not np.isnan(iou_val):
                    iou_list.append(iou_val)
            except (ValueError, TypeError):
                pass

    if iou_list:
        mean_iou = np.mean(iou_list)
        print(f"==> Mean IoU: {round(mean_iou * 100, 2)}")
    else:
        print("==> Mean IoU: N/A (no valid 'iou' field found)")

def main():
    parser = argparse.ArgumentParser(description="Evaluate TreeBench results.")
    parser.add_argument("--answer-file", type=str, required=True,
                        help="Path to the answer file in JSONL format.")
    args = parser.parse_args()
    eval_treebench(args.answer_file)

if __name__ == "__main__":
    main()