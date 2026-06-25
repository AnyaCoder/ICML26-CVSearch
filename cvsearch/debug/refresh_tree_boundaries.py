import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cvsearch.debug.image_io import debug_artifact_path, existing_debug_image
from cvsearch.debug.recorder import CVSearchDebugRecorder
from cvsearch.models.modeling_sam3 import ConstrainedTreeBuilder, sam3_inference
from cvsearch.utils import release_cuda_cache

PROJECT_ROOT = Path(__file__).resolve().parents[2]


warnings.filterwarnings("ignore")


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_path(root, path):
    path = Path(path)
    if path.is_absolute():
        return path
    return root / path


def normalize_prompts(value):
    if isinstance(value, list):
        prompts = [str(item).strip() for item in value if str(item).strip()]
    elif value:
        prompts = [str(value).strip()]
    else:
        prompts = []
    return prompts or ["object"]


def extract_feature_map(sam_model, image, prompts):
    with torch.inference_mode():
        backbone_out, _, _ = sam_model.batch_inference(image, prompts)
    image_features = backbone_out["vision_features"]
    if isinstance(image_features, torch.Tensor):
        feat = image_features.detach().cpu().float().numpy()
    else:
        feat = np.asarray(image_features, dtype=np.float32)
    del backbone_out
    del image_features
    release_cuda_cache()
    return feat.squeeze(0)


def use_cuda_device(device: str) -> None:
    parsed = torch.device(device)
    if parsed.type == "cuda" and parsed.index is not None:
        torch.cuda.set_device(parsed.index)


def build_and_record(sample_dir, name, image, feat, max_depth, use_local_normalization=True):
    builder = ConstrainedTreeBuilder(
        feature_map=feat,
        n_atoms=600,
        pos_weight=3.5,
        split_threshold=0.3,
        keep_threshold=0.15,
        use_local_normalization=use_local_normalization,
        use_silhouette_score=True,
    )
    tree = builder.build_tree(max_depth=max_depth, min_splits=4, max_splits=8)
    recorder = CVSearchDebugRecorder(sample_dir)
    recorder.record_tree_boundaries(name, image, builder, tree)


def iter_existing_second_tree_crops(sample_dir):
    prefix = "04_second_tree_"
    suffix = "_tree.json"
    for tree_json in sorted(sample_dir.glob(f"{prefix}*{suffix}")):
        target_slug = tree_json.name[len(prefix) : -len(suffix)]
        crop_path = debug_artifact_path(sample_dir, f"05_second_{target_slug}_crop")
        if crop_path.exists():
            yield target_slug, crop_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", default=PROJECT_ROOT)
    parser.add_argument(
        "--answers-file",
        default="cvsearch/CVsearch/eval/answers/vstar/Qwen2.5-VL-7B-Instruct/cvsearch_wrong_rerun.jsonl",
    )
    parser.add_argument("--image-root", default="datasets/hr_data/vstar")
    parser.add_argument("--sam-model-path", default="models/facebook/sam3/sam3.pt")
    parser.add_argument("--sam-device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--indices", default=None, help="Comma-separated zero-based row indices in answers-file.")
    parser.add_argument("--skip-second", action="store_true")
    args = parser.parse_args()

    root = Path(args.root_path).resolve()
    answers_file = resolve_path(root, args.answers_file)
    image_root = resolve_path(root, args.image_root)
    sam_model_path = resolve_path(root, args.sam_model_path)
    rows = load_jsonl(answers_file)

    selected = list(range(len(rows)))
    if args.indices:
        selected = [int(item.strip()) for item in args.indices.split(",") if item.strip()]
    if args.limit is not None:
        selected = selected[: args.limit]

    print(f"answers_file={answers_file}")
    print(f"selected={len(selected)}")
    print(f"sam_device={args.sam_device}")
    use_cuda_device(args.sam_device)
    sam_model = sam3_inference(model_path=str(sam_model_path))

    for ordinal, row_idx in enumerate(selected, start=1):
        row = rows[row_idx]
        sample_dir = resolve_path(root, row["debug_dir"])
        primary_image_path = existing_debug_image(sample_dir / "00_original.jpg")
        if not primary_image_path.exists():
            primary_image_path = image_root / row["input_image"]
        image = Image.open(primary_image_path).convert("RGB")
        prompts = normalize_prompts(row.get("targets") or row.get("target_object"))

        print(f"[{ordinal}/{len(selected)}] primary {sample_dir.name} prompts={prompts}")
        feat = extract_feature_map(sam_model, image, prompts)
        build_and_record(sample_dir, "primary_tree", image, feat, max_depth=3)
        del feat
        release_cuda_cache()

        if args.skip_second:
            continue
        for target_slug, crop_path in iter_existing_second_tree_crops(sample_dir):
            crop = Image.open(crop_path).convert("RGB")
            prompt = target_slug.replace("_", " ")
            print(f"[{ordinal}/{len(selected)}] second {sample_dir.name}/{target_slug}")
            feat_sub = extract_feature_map(sam_model, crop, [prompt])
            build_and_record(
                sample_dir,
                f"second_tree_{target_slug}",
                crop,
                feat_sub,
                max_depth=2,
                use_local_normalization=True,
            )
            del feat_sub
            release_cuda_cache()

    print("refresh_tree_boundaries_done")


if __name__ == "__main__":
    main()
