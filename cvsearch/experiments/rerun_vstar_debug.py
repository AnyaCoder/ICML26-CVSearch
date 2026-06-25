import argparse
import copy
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import spacy
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CVSEARCH_ROOT = PROJECT_ROOT / "cvsearch"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CVSEARCH_ROOT))

from cvsearch.debug.recorder import CVSearchDebugRecorder
from cvsearch.evidence_memory import (
    AttentionBoxGrounder,
    AttentionGuidedWindowBuilder,
    EvidenceLayoutConfig,
    EvidenceMemoryCompiler,
    EvidenceRetentionConfig,
    GroundingDINOBoxVerifier,
    PerTargetEvidenceLayout,
    VerifierFirstEvidenceKeeper,
    WindowBuilderConfig,
)
from cvsearch.evidence_memory.qwen_attention_provider import QwenFilteredAttentionConfig, QwenFilteredAttentionProvider
from CVSearch import get_cvsearch_response
from models.modeling_qwenvl import ModelQwenVL
from models.modeling_sam3 import sam3_inference


warnings.filterwarnings("ignore")
_original_np_load = np.load


def _patched_np_load(*args, **kwargs):
    kwargs["allow_pickle"] = True
    return _original_np_load(*args, **kwargs)


np.load = _patched_np_load


def safe_slug(text, max_len=80):
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in "._-":
            keep.append(ch)
        else:
            keep.append("_")
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return (slug or "sample")[:max_len]


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_search_kwargs():
    def pop_limit_func(max_depth):
        return max_depth * 3

    return {
        "pop_limit": pop_limit_func,
        "threshold_descrease": [0.05, 0.1, 0.2],
        "answering_confidence_threshold_lower": 0,
        "answering_confidence_threshold_upper": 0.9,
        "fast_threshold": 0.6,
    }


def use_cuda_device(device: str) -> None:
    parsed = torch.device(device)
    if parsed.type == "cuda" and parsed.index is not None:
        torch.cuda.set_device(parsed.index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", default=PROJECT_ROOT)
    parser.add_argument("--benchmark", default="vstar")
    parser.add_argument("--model-path", default="models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--annotation-path", default="datasets/hr_data")
    parser.add_argument("--nlp-model-path", default="models/en_core_web_sm-3.8.0")
    parser.add_argument("--sam-model-path", default="models/facebook/sam3/sam3.pt")
    parser.add_argument("--sam-device", default="cuda:1")
    parser.add_argument("--vlm-device", default="cuda:0")
    parser.add_argument(
        "--source-answers-file",
        default="CVsearch/eval/answers/vstar/Qwen2.5-VL-7B-Instruct/cvsearch.jsonl",
    )
    parser.add_argument(
        "--answers-file",
        default="CVsearch/eval/answers/vstar/Qwen2.5-VL-7B-Instruct/cvsearch_wrong_rerun.jsonl",
    )
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--indices", default=None, help="Comma-separated zero-based source indices.")
    parser.add_argument("--selection", choices=["wrong", "correct", "all"], default="wrong")
    parser.add_argument("--test-types", default=None, help="Comma-separated annotation test_type values.")
    parser.add_argument("--enable-evidence-memory", action="store_true")
    parser.add_argument("--evidence-device", default=None)
    parser.add_argument("--dino-model-path", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--dino-device", default=None)
    parser.add_argument("--dino-threshold", type=float, default=0.25)
    parser.add_argument("--dino-text-threshold", type=float, default=0.25)
    parser.add_argument("--dino-local-files-only", action="store_true")
    parser.add_argument("--dino-max-batch-items", type=int, default=1)
    parser.add_argument("--max-batch-visual-tokens", type=int, default=None)
    parser.add_argument("--max-batch-items", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.root_path).resolve()
    benchmark = args.benchmark
    model_path = root / args.model_path
    annotation_root = root / args.annotation_path
    image_folder = annotation_root / benchmark
    source_answers_file = root / "cvsearch" / args.source_answers_file
    answers_file = root / "cvsearch" / args.answers_file
    answers_file.parent.mkdir(parents=True, exist_ok=True)

    if args.debug_dir:
        debug_root = Path(args.debug_dir)
        if not debug_root.is_absolute():
            debug_root = root / args.debug_dir
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        debug_root = root / "debug" / f"vstar_wrong_qwen25vl7b_{stamp}"
    debug_root.mkdir(parents=True, exist_ok=True)

    original_answers = load_jsonl(source_answers_file)
    annotations = load_json(annotation_root / benchmark / f"annotation_{benchmark}.json")
    ic_examples = load_json(root / "cvsearch" / "ic_examples" / f"{benchmark}.json")

    if len(original_answers) != len(annotations):
        raise RuntimeError(
            f"Source answer length {len(original_answers)} != annotation length {len(annotations)}"
        )

    selected = []
    explicit_indices = None
    if args.indices:
        explicit_indices = {int(item.strip()) for item in args.indices.split(",") if item.strip()}
    allowed_test_types = None
    if args.test_types:
        allowed_test_types = {item.strip() for item in args.test_types.split(",") if item.strip()}

    for idx, item in enumerate(original_answers):
        annotation = annotations[idx]
        if allowed_test_types is not None and annotation.get("test_type") not in allowed_test_types:
            continue
        if explicit_indices is not None:
            if idx in explicit_indices:
                selected.append(idx)
        elif args.selection == "all":
            selected.append(idx)
        elif args.selection == "correct" and item.get("output") == 0:
            selected.append(idx)
        elif args.selection == "wrong" and item.get("output") != 0:
            selected.append(idx)

    if args.limit is not None:
        selected = selected[: args.limit]

    print(f"source_answers_file={source_answers_file}")
    print(f"answers_file={answers_file}")
    print(f"debug_root={debug_root}")
    print(f"selected_wrong_samples={len(selected)} indices={selected}")

    search_model = ModelQwenVL(
        model_path=str(model_path),
        device=args.vlm_device,
        torch_dtype=torch.bfloat16,
        patch_scale=1.2,
    )
    nlp = spacy.load(name=str(root / args.nlp_model_path))
    use_cuda_device(args.sam_device)
    sam3 = sam3_inference(model_path=str(root / args.sam_model_path))
    evidence_compiler = build_evidence_compiler(args, model_path, search_model) if args.enable_evidence_memory else None

    search_kwargs = build_search_kwargs()
    decomposed_question_template = "What is the appearance of the {}?"
    summary = {
        "source_answers_file": str(source_answers_file),
        "answers_file": str(answers_file),
        "debug_root": str(debug_root),
        "selected_indices": selected,
        "search_kwargs": {
            k: v for k, v in search_kwargs.items() if k != "pop_limit"
        },
        "sam_device": args.sam_device,
        "vlm_device": args.vlm_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "results": [],
    }

    with open(answers_file, "w") as out:
        for idx in tqdm(selected):
            source_item = original_answers[idx]
            annotation = copy.deepcopy(annotations[idx])
            sample_id = idx + 1
            sample_slug = safe_slug(
                f"sample_{sample_id:03d}_{annotation.get('test_type')}_{Path(annotation.get('input_image', '')).stem}"
            )
            sample_dir = debug_root / sample_slug
            recorder = CVSearchDebugRecorder(sample_dir, sample_index=sample_id)

            response = get_cvsearch_response(
                sam_model=sam3,
                zoom_model=search_model,
                nlp_model=nlp,
                annotation=annotation,
                ic_examples=ic_examples,
                decomposed_question_template=decomposed_question_template,
                image_folder=str(image_folder),
                debug_recorder=recorder,
                evidence_compiler=evidence_compiler,
                **search_kwargs,
            )
            annotation["output"] = response
            annotation["original_index"] = idx
            annotation["original_output"] = source_item.get("output")
            annotation["debug_dir"] = str(sample_dir)
            out.write(json.dumps(annotation, ensure_ascii=False) + "\n")
            out.flush()

            summary["results"].append(
                {
                    "original_index": idx,
                    "input_image": annotation.get("input_image"),
                    "test_type": annotation.get("test_type"),
                    "old_output": source_item.get("output"),
                    "new_output": response,
                    "is_correct": response == 0,
                    "debug_dir": str(sample_dir),
                }
            )
            with open(debug_root / "summary.json", "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

    total = len(summary["results"])
    fixed = sum(1 for item in summary["results"] if item["is_correct"])
    summary["fixed_count"] = fixed
    summary["remaining_wrong_count"] = total - fixed
    with open(debug_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"wrong_rerun_done total={total} fixed={fixed} remaining_wrong={total - fixed}")
    print(f"summary={debug_root / 'summary.json'}")


def build_evidence_compiler(args: argparse.Namespace, model_path: Path, search_model=None) -> EvidenceMemoryCompiler:
    device = args.evidence_device or args.vlm_device
    attention_provider = QwenFilteredAttentionProvider(
        QwenFilteredAttentionConfig(
            model_path=str(model_path),
            device=device,
            dtype=torch.bfloat16,
            max_batch_visual_tokens=args.max_batch_visual_tokens,
            max_batch_items=args.max_batch_items,
        ),
        external_model=getattr(search_model, "model", None),
        external_processor=getattr(search_model, "processor", None),
    )
    dino_verifier = GroundingDINOBoxVerifier(
        model_path=args.dino_model_path,
        device=args.dino_device or device,
        threshold=args.dino_threshold,
        text_threshold=args.dino_text_threshold,
        region="attention_box",
        local_files_only=args.dino_local_files_only,
        max_batch_items=args.dino_max_batch_items,
    )
    return EvidenceMemoryCompiler(
        window_builder=AttentionGuidedWindowBuilder(
            attention_provider,
            config=WindowBuilderConfig(
                attention_analysis_min_size=1.0,
                attention_analysis_scale=1.0,
                attention_min_size=112.0,
                attention_margin=1.0,
                moment_beta=1.0,
            ),
        ),
        keeper=VerifierFirstEvidenceKeeper(
            verifier=dino_verifier,
            grounder=AttentionBoxGrounder(),
            config=EvidenceRetentionConfig(
                max_items_per_target=9,
                max_total_items=9,
                attention_nms_iou_threshold=0.8,
            ),
        ),
        layout=PerTargetEvidenceLayout(
            EvidenceLayoutConfig(
                top_k_per_target=9,
                montage_mode="original_merge",
            )
        ),
    )


if __name__ == "__main__":
    main()
