import argparse
import json
from tqdm import tqdm
import numpy as np


def flatten_and_sum(val_list):
    """
    - [1] -> 1
    - [[2, 3]] -> 5
    - [1, [2, 3]] -> 6
    """
    total = 0
    for item in val_list:
        if isinstance(item, list):
            total += sum(item)
        else:
            total += item
    return total

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
        test_type = info['test_type']
        if test_type not in all_acc:
            all_acc[test_type] = []
            all_pop[test_type] = []
            all_zoom_in[test_type] = []
            all_zoom_out[test_type] = []

        all_acc[test_type].append(info['output'] == 0)
        total_pop = flatten_and_sum(info.get('num_pop', []))
        total_zoom_in = flatten_and_sum(info.get('num_zoom_in', []))
        total_zoom_out = flatten_and_sum(info.get('num_zoom_out', []))

        all_pop[test_type].append(total_pop)
        all_zoom_in[test_type].append(total_zoom_in)
        all_zoom_out[test_type].append(total_zoom_out)

    total_acc = []
    total_pop = []
    total_zoom_in = []
    total_zoom_out = []

    print("\n" + "=" * 50)
    for test_type in all_acc:
        print(f"Test Type: {test_type}")
        acc_val = 100 * np.mean(all_acc[test_type])
        print(f"acc:      {acc_val:.2f}%")
        total_acc.extend(all_acc[test_type])
        print('=' * 50)

    print("Overall Summary:")
    print(f"Total Acc:      {100 * np.mean(total_acc):.2f}%")