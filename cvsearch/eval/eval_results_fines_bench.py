import argparse
import json
import re
import difflib
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QA results for Option and Reasoning tasks.")
    parser.add_argument("--question-type", type=str, choices=["option", "reasoning", "all"], default="all",
                        help="The type of question to evaluate.")
    parser.add_argument("--answer-file", type=str, required=True,
                        help="Path to the JSON file containing model outputs and ground truth.")
    return parser.parse_args()

def init_metric_dict():
    return {
        "colors": {"num": 0, "correct": 0},
        "shape": {"num": 0, "correct": 0},
        "others": {"num": 0, "correct": 0},
        "position": {"num": 0, "correct": 0},
        "avg": {"num": 0, "correct": 0}
    }

def update_metric(metric_dict, attribute, is_correct):
    if attribute not in metric_dict and attribute != "avg":
        attribute = "others"

    metric_dict[attribute]["num"] += 1
    if is_correct:
        metric_dict[attribute]["correct"] += 1

    metric_dict["avg"]["num"] += 1
    if is_correct:
        metric_dict["avg"]["correct"] += 1

    return metric_dict


def eval_option(final_response, gt_answer):
    if final_response is None: final_response = ""
    if gt_answer is None: gt_answer = ""

    final_response = str(final_response).lower()
    gt_answer = str(gt_answer).lower()

    pattern = rf"(?:\b|[\(\[\{{'\" ]){re.escape(gt_answer)}(?:\b|[\)\]\}}'\" ,.!?])"

    if re.search(pattern, final_response):
        return True
    return False

def eval_reasoning(final_response, gt_answer):
    if final_response is None: final_response = ""
    if gt_answer is None: gt_answer = ""

    final_response = str(final_response).lower()
    gt_answer = str(gt_answer).lower()

    response_words = final_response.split()

    for word in response_words:
        similarity = difflib.SequenceMatcher(None, word, gt_answer).ratio()
        if similarity >= 0.8:
            return True

    return False

def print_metrics(metrics, title):
    print(f"\n{'=' * 20} {title} Results {'=' * 20}")
    print(f"{'Attribute':<15} | {'Total':<10} | {'Correct':<10} | {'Accuracy':<10}")
    print("-" * 55)

    keys = [k for k in metrics.keys() if k != "avg"] + ["avg"]

    for key in keys:
        data = metrics[key]
        num = data["num"]
        correct = data["correct"]
        acc = (correct / num) * 100 if num > 0 else 0.0
        print(f"{key:<15} | {num:<10} | {correct:<10} | {acc:.2f}%")

def main():
    args = parse_args()
    data_list = []
    try:
        with open(args.answer_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith("["):
                data_list = json.loads(content)
            else:
                f.seek(0)
                for line in f:
                    if line.strip():
                        data_list.append(json.loads(line))
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    metrics = {
        "option": init_metric_dict(),
        "reasoning": init_metric_dict()
    }

    print(f"Start evaluating {len(data_list)} samples...")

    for item in data_list:
        q_type = item.get("Q-type")

        if args.question_type != "all" and q_type != args.question_type:
            continue
        if q_type not in ["option", "reasoning"]:
            continue

        gt_answer = item.get("A")
        output = item.get("output")
        attribute = item.get("attribute", "others")

        if q_type == "option":
            is_correct = eval_option(output, gt_answer)
            metrics["option"] = update_metric(metrics["option"], attribute, is_correct)

        elif q_type == "reasoning":
            is_correct = eval_reasoning(output, gt_answer)
            metrics["reasoning"] = update_metric(metrics["reasoning"], attribute, is_correct)

    if args.question_type in ["option", "all"] and metrics["option"]["avg"]["num"] > 0:
        print_metrics(metrics["option"], "Option (Multiple Choice)")

    if args.question_type in ["reasoning", "all"] and metrics["reasoning"]["avg"]["num"] > 0:
        print_metrics(metrics["reasoning"], "Reasoning (Free Form)")


if __name__ == "__main__":
    main()