import argparse
import json
from tqdm import tqdm
import numpy as np


def flatten_list(nested_list):
    """
    [1, [2, 3], [[4]]] -> [1, 2, 3, 4]
    """
    flat = []
    for item in nested_list:
        if isinstance(item, list):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=str, required=True)
    args = parser.parse_args()

    infos = []
    with open(args.answers_file, 'r') as f:
        for line in f:
            infos.append(json.loads(line))
    print("len:", len(infos))

    all_acc = {}
    all_pop = {}
    all_zoom_in = {}
    all_zoom_out = {}

    for info in tqdm(infos):
        if info['category'] not in all_acc:
            all_acc[info['category']] = []
            all_pop[info['category']] = []
            all_zoom_in[info['category']] = []
            all_zoom_out[info['category']] = []

        for ans, cho in zip(info['answer'], info['output']):
            x = ''
            if len(cho) == 1:
                x = cho[0]
            else:
                for c in cho:
                    if c in ['A', 'B', 'C', 'D']:
                        x = c
                        break
            all_acc[info['category']].append(ans == x)

        raw_pop = info.get('num_pop', [0])
        raw_zoom_in = info.get('num_zoom_in', [0])
        raw_zoom_out = info.get('num_zoom_out', [0])

        all_pop[info['category']].extend(flatten_list(raw_pop))
        all_zoom_in[info['category']].extend(flatten_list(raw_zoom_in))
        all_zoom_out[info['category']].extend(flatten_list(raw_zoom_out))

    total_acc = []
    total_pop = []
    total_zoom_in = []
    total_zoom_out = []

    print("\n" + "=" * 50)
    for category in all_acc:
        print(f"Category: {category}")

        acc_val = 100 * np.mean(all_acc[category])
        print(f"acc: {acc_val:.2f}")

        if len(all_pop[category]) == 0: all_pop[category] = [0]
        if len(all_zoom_in[category]) == 0: all_zoom_in[category] = [0]
        if len(all_zoom_out[category]) == 0: all_zoom_out[category] = [0]

        pop_val = np.mean(all_pop[category])
        zoom_in_val = np.mean(all_zoom_in[category])
        zoom_out_val = np.mean(all_zoom_out[category])

        print(f"pop: {pop_val:.4f}")
        print(f"zoom_in: {zoom_in_val:.4f}")
        print(f"zoom_out: {zoom_out_val:.4f}")

        total_acc.extend(all_acc[category])
        total_pop.extend(all_pop[category])
        total_zoom_in.extend(all_zoom_in[category])
        total_zoom_out.extend(all_zoom_out[category])
        print('=' * 50)

    print("Total Summary:")
    print(f"total acc: {100 * np.mean(total_acc):.2f}")
    print(f"total pop: {np.mean(total_pop):.4f}")
    print(f"total zoom_in: {np.mean(total_zoom_in):.4f}")
    print(f"total zoom_out: {np.mean(total_zoom_out):.4f}")