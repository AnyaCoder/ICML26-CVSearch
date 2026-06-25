import json
import math
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.segmentation import slic


def _safe_slug(text, max_len=48):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return (text or "item")[:max_len]


def _xywh_to_xyxy(bbox):
    x, y, w, h = [int(v) for v in bbox]
    return [x, y, x + w, y + h]


def _clip_xyxy(box, image):
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = image.size
    return [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]


def _draw_label(draw, xy, text, fill):
    font = ImageFont.load_default()
    x, y = xy
    box = draw.textbbox((x, y), text, font=font)
    pad = 2
    bg = [box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad]
    draw.rectangle(bg, fill=(255, 255, 255))
    draw.text((x, y), text, fill=fill, font=font)


def _draw_box(draw, box, color, label=None, width=4):
    x1, y1, x2, y2 = [int(v) for v in box]
    for i in range(width):
        draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)
    if label:
        _draw_label(draw, (x1 + 2, max(0, y1 - 13)), label, color)


def _fit_thumb(image, max_size=(260, 180)):
    thumb = image.copy()
    thumb.thumbnail(max_size)
    return thumb


class CVSearchDebugRecorder:
    def __init__(self, sample_dir, sample_index=None, max_tree_nodes=48, max_trace_crops=36):
        self.sample_dir = Path(sample_dir)
        self.sample_index = sample_index
        self.max_tree_nodes = max_tree_nodes
        self.max_trace_crops = max_trace_crops
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.original_image = None
        self.annotation = None
        self.trace_counter = 0
        self.traces = {}
        self.events = []

    def _path(self, name):
        return self.sample_dir / name

    def write_json(self, name, data):
        with open(self._path(name), "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def start_sample(self, annotation, image_pil, image_path):
        self.annotation = dict(annotation)
        self.original_image = image_pil.copy()
        image_pil.save(self._path("00_original.png"))
        self.write_json(
            "00_metadata.json",
            {
                "sample_index": self.sample_index,
                "image_path": image_path,
                "question": annotation.get("question"),
                "options": annotation.get("options"),
                "target_object": annotation.get("target_object"),
                "gt_bbox_xywh": annotation.get("bbox"),
                "input_image": annotation.get("input_image"),
                "test_type": annotation.get("test_type"),
            },
        )
        self._save_gt_overlay(annotation, image_pil)

    def _save_gt_overlay(self, annotation, image_pil):
        out = image_pil.copy()
        draw = ImageDraw.Draw(out)
        targets = annotation.get("target_object") or []
        for i, bbox in enumerate(annotation.get("bbox") or []):
            label = f"GT {i + 1}"
            if i < len(targets):
                label += f": {targets[i]}"
            _draw_box(draw, _xywh_to_xyxy(bbox), (0, 180, 0), label, width=5)
        self._draw_question_band(out, annotation)
        out.save(self._path("01_gt_boxes.png"))

    def _draw_question_band(self, image, annotation):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        lines = [f"#{self.sample_index}" if self.sample_index is not None else ""]
        lines.append(annotation.get("question", ""))
        options = annotation.get("options") or []
        for i, option in enumerate(options):
            prefix = chr(ord("A") + i)
            lines.append(f"{prefix}. {option}")
        lines = [line for line in lines if line]
        if not lines:
            return
        text_h = 12 * len(lines) + 10
        draw.rectangle([0, 0, image.size[0], text_h], fill=(255, 255, 255))
        y = 4
        for line in lines:
            draw.text((6, y), line[:180], fill=(0, 0, 0), font=font)
            y += 12

    def record_root_confidence(self, confidence):
        self.events.append({"event": "root_confidence", "confidence": float(confidence)})
        self.write_json("02_root_confidence.json", self.events[-1])

    def record_sam(self, name, image_pil, text_targets, processed_results, target_ids, sam_success_flags, sam_bboxes, offset=(0, 0)):
        event_slug = _safe_slug(name)
        offset_x, offset_y = offset
        local = image_pil.copy()
        local_draw = ImageDraw.Draw(local)
        global_img = self.original_image.copy() if self.original_image is not None else None
        global_draw = ImageDraw.Draw(global_img) if global_img is not None else None
        records = []
        colors = [(220, 30, 30), (30, 100, 220), (220, 140, 0), (130, 40, 180), (0, 150, 130)]
        for pos, t_id in enumerate(target_ids):
            color = colors[pos % len(colors)]
            target = text_targets[pos] if pos < len(text_targets) else f"target_{t_id}"
            item = {"target_id": int(t_id), "target": target, "boxes_xyxy": [], "scores": []}
            boxes = processed_results[t_id]["boxes"].float().cpu().numpy()
            scores = processed_results[t_id]["scores"].float().cpu().numpy()
            for box_idx, box in enumerate(boxes):
                box = [int(v) for v in box[:4]]
                score = float(scores[box_idx]) if box_idx < len(scores) else None
                label = f"SAM {pos + 1}:{target}"
                if score is not None:
                    label += f" {score:.2f}"
                _draw_box(local_draw, _clip_xyxy(box, local), color, label, width=3)
                if global_draw is not None:
                    shifted = [box[0] + offset_x, box[1] + offset_y, box[2] + offset_x, box[3] + offset_y]
                    _draw_box(global_draw, _clip_xyxy(shifted, global_img), color, label, width=3)
                item["boxes_xyxy"].append(box)
                item["scores"].append(score)
            records.append(item)

        local.save(self._path(f"03_{event_slug}_local.png"))
        if global_img is not None and offset != (0, 0):
            global_img.save(self._path(f"03_{event_slug}_global.png"))
        self.write_json(
            f"03_{event_slug}.json",
            {
                "targets": list(text_targets),
                "target_ids": [int(t) for t in target_ids],
                "sam_success_flags": [int(v) for v in sam_success_flags],
                "sam_bboxes_xyxy": sam_bboxes,
                "offset": [int(offset_x), int(offset_y)],
                "raw": records,
            },
        )

    def record_tree(self, name, image_tree):
        event_slug = _safe_slug(name)
        by_depth = {}
        queue = [image_tree.root]
        while queue:
            node = queue.pop(0)
            by_depth.setdefault(node.depth, []).append(node)
            queue.extend(node.children)
        summary = {}
        for depth, nodes in sorted(by_depth.items()):
            if depth == 0:
                continue
            nodes_sorted = sorted(nodes, key=lambda n: getattr(n, "prior_prob", 0), reverse=True)
            summary[str(depth)] = [
                {
                    "id": getattr(node, "id", None),
                    "bbox_xywh": list(map(int, node.state.bbox)),
                    "prior_prob": float(getattr(node, "prior_prob", 0)),
                    "relative_score": float(getattr(node, "relative_score", 0)),
                    "complexity": float(getattr(node, "complexity", 0)),
                }
                for node in nodes_sorted
            ]
            self._save_tree_overlay(event_slug, depth, image_tree.image_pil, nodes_sorted[: self.max_tree_nodes])
            self._save_node_crop_grid(
                f"04_{event_slug}_depth{depth}_crops.png",
                image_tree.image_pil,
                nodes_sorted[: min(self.max_tree_nodes, 24)],
                title=f"{name} depth {depth} top crops",
            )
        self.write_json(f"04_{event_slug}_tree.json", summary)

    def record_tree_boundaries(self, name, image_pil, builder, tree_dict):
        event_slug = _safe_slug(name)
        atom_labels = np.asarray(builder.atom_labels)
        self._save_feature_pca_slic(event_slug, image_pil, builder)
        self._save_boundary_overlay(
            f"04_{event_slug}_slic_superpixels.png",
            image_pil,
            atom_labels,
            title="Feature-map SLIC atoms used by tree",
        )
        self._save_boundary_overlay(
            f"04_{event_slug}_feature_slic_atoms.png",
            image_pil,
            atom_labels,
            title="Feature-map SLIC atoms used by tree",
        )
        self._save_rgb_slic_superpixels(event_slug, image_pil)

        max_depth = self._tree_max_depth(tree_dict)
        for depth in range(1, max_depth + 1):
            nodes = []
            self._collect_tree_nodes_at_depth(tree_dict, depth, nodes)
            if not nodes:
                continue
            cluster_map = self._cluster_map_from_nodes(atom_labels, nodes)
            self._save_boundary_overlay(
                f"04_{event_slug}_depth{depth}_overlay.png",
                image_pil,
                cluster_map,
                title=f"{name} depth {depth} semantic clustering",
            )
            self._save_cluster_color_map(
                f"04_{event_slug}_depth{depth}_semantic_map.png",
                image_pil,
                cluster_map,
                title=f"{name} depth {depth} semantic map",
            )

    def _tree_max_depth(self, node):
        children = node.get("children") or []
        if not children:
            return int(node.get("depth", 0))
        return max(self._tree_max_depth(child) for child in children)

    def _collect_tree_nodes_at_depth(self, node, depth, out):
        if int(node.get("depth", 0)) == depth:
            out.append(node)
            return
        for child in node.get("children") or []:
            self._collect_tree_nodes_at_depth(child, depth, out)

    def _cluster_map_from_nodes(self, atom_labels, nodes):
        max_label = int(atom_labels.max()) if atom_labels.size else 0
        atom_to_cluster = np.full(max_label + 1, -1, dtype=np.int32)
        nodes_sorted = sorted(nodes, key=lambda n: str(n.get("node_id", "")))
        for cluster_id, node in enumerate(nodes_sorted):
            atom_indices = np.asarray(node.get("atom_indices", []), dtype=np.int64)
            atom_indices = atom_indices[(atom_indices >= 0) & (atom_indices <= max_label)]
            atom_to_cluster[atom_indices] = cluster_id
        return atom_to_cluster[atom_labels]

    def _resize_label_map(self, label_map, size):
        label_map = np.asarray(label_map)
        if label_map.size == 0:
            return label_map
        min_label = label_map.min()
        shifted = label_map - min_label
        if shifted.max() <= np.iinfo(np.uint16).max:
            img = Image.fromarray(shifted.astype(np.uint16))
            resized = img.resize(size, Image.Resampling.NEAREST)
            return np.asarray(resized).astype(np.int32) + int(min_label)
        img = Image.fromarray(shifted.astype(np.int32), mode="I")
        resized = img.resize(size, Image.Resampling.NEAREST)
        return np.asarray(resized).astype(np.int32) + int(min_label)

    def _boundary_mask(self, label_map, width=2):
        labels = np.asarray(label_map)
        boundary = np.zeros(labels.shape, dtype=bool)
        valid = labels >= 0
        horizontal = (labels[:, 1:] != labels[:, :-1]) & valid[:, 1:] & valid[:, :-1]
        vertical = (labels[1:, :] != labels[:-1, :]) & valid[1:, :] & valid[:-1, :]
        boundary[:, 1:] |= horizontal
        boundary[:, :-1] |= horizontal
        boundary[1:, :] |= vertical
        boundary[:-1, :] |= vertical
        for _ in range(max(0, width - 1)):
            padded = np.pad(boundary, 1, mode="edge")
            boundary = (
                padded[1:-1, 1:-1]
                | padded[:-2, 1:-1]
                | padded[2:, 1:-1]
                | padded[1:-1, :-2]
                | padded[1:-1, 2:]
            )
            boundary &= valid
        return boundary & valid

    def _draw_title(self, image, title):
        if not title:
            return
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        text = title[:140]
        box = draw.textbbox((6, 6), text, font=font)
        draw.rectangle([box[0] - 3, box[1] - 3, box[2] + 3, box[3] + 3], fill=(255, 255, 255))
        draw.text((6, 6), text, fill=(0, 0, 0), font=font)

    def _save_boundary_overlay(self, filename, image_pil, label_map, title=None, color=(0, 0, 0)):
        labels = self._resize_label_map(label_map, image_pil.size)
        if labels.size == 0:
            return
        out = image_pil.copy().convert("RGB")
        arr = np.asarray(out).copy()
        boundary = self._boundary_mask(labels, width=2)
        arr[boundary] = color
        out = Image.fromarray(arr)
        self._draw_title(out, title)
        out.save(self._path(filename))

    def _save_cluster_color_map(self, filename, image_pil, label_map, title=None):
        labels = self._resize_label_map(label_map, image_pil.size)
        if labels.size == 0:
            return
        base = np.asarray(image_pil.convert("RGB")).astype(np.float32)
        valid = labels >= 0
        color_table = np.array(
            [
                [230, 25, 75],
                [60, 180, 75],
                [0, 130, 200],
                [245, 130, 48],
                [145, 30, 180],
                [70, 240, 240],
                [240, 50, 230],
                [210, 245, 60],
                [250, 190, 190],
                [0, 128, 128],
                [230, 190, 255],
                [170, 110, 40],
            ],
            dtype=np.float32,
        )
        colors = np.zeros_like(base)
        safe_labels = np.maximum(labels, 0)
        colors[valid] = color_table[safe_labels[valid] % len(color_table)]
        blended = base.copy()
        blended[valid] = base[valid] * 0.62 + colors[valid] * 0.38
        boundary = self._boundary_mask(labels, width=2)
        blended[boundary] = [0, 0, 0]
        out = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
        self._draw_title(out, title)
        out.save(self._path(filename))

    def _save_feature_pca_slic(self, event_slug, image_pil, builder):
        feat = np.asarray(builder.feat, dtype=np.float32)
        if feat.ndim != 3:
            return
        c, h, w = feat.shape
        pixels = feat.reshape(c, -1).T
        pixels = pixels - pixels.mean(axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(pixels, full_matrices=False)
            rgb = pixels @ vt[:3].T
        except np.linalg.LinAlgError:
            return
        rgb = rgb.reshape(h, w, 3)
        lo = np.percentile(rgb, 1, axis=(0, 1), keepdims=True)
        hi = np.percentile(rgb, 99, axis=(0, 1), keepdims=True)
        rgb = (rgb - lo) / (hi - lo + 1e-6)
        rgb = np.clip(rgb, 0, 1)
        pca_img = Image.fromarray((rgb * 255).astype(np.uint8)).resize(image_pil.size, Image.Resampling.BILINEAR)
        labels = self._resize_label_map(builder.atom_labels, image_pil.size)
        arr = np.asarray(pca_img).copy()
        arr[self._boundary_mask(labels, width=2)] = [0, 0, 0]
        out = Image.fromarray(arr)
        self._draw_title(out, "PCA feature map + SLIC boundaries")
        out.save(self._path(f"04_{event_slug}_feature_pca_slic.png"))

    def _save_rgb_slic_superpixels(self, event_slug, image_pil):
        work = image_pil.convert("RGB")
        work.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
        image_arr = np.asarray(work, dtype=np.float32) / 255.0
        labels = slic(
            image_arr,
            n_segments=700,
            compactness=8,
            sigma=1,
            start_label=0,
            channel_axis=2,
            convert2lab=True,
        )
        self._save_boundary_overlay(
            f"04_{event_slug}_rgb_slic_superpixels.png",
            image_pil,
            labels,
            title="RGB SLIC superpixels for visual reference",
        )

    def _save_tree_overlay(self, event_slug, depth, image_pil, nodes):
        out = image_pil.copy()
        draw = ImageDraw.Draw(out)
        palette = [(230, 50, 50), (50, 130, 230), (240, 150, 30), (120, 60, 210)]
        for rank, node in enumerate(nodes):
            color = palette[rank % len(palette)]
            label = f"{rank + 1}:{getattr(node, 'id', '?')} p={getattr(node, 'prior_prob', 0):.2f}"
            _draw_box(draw, _xywh_to_xyxy(node.state.bbox), color, label, width=2)
        out.save(self._path(f"04_{event_slug}_depth{depth}_overlay.png"))

    def _save_node_crop_grid(self, filename, image_pil, nodes, title):
        items = []
        for rank, node in enumerate(nodes):
            box = _clip_xyxy(_xywh_to_xyxy(node.state.bbox), image_pil)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            crop = image_pil.crop(box)
            label = f"{rank + 1} id={getattr(node, 'id', '?')} d={getattr(node, 'depth', '?')}"
            label += f" p={getattr(node, 'prior_prob', 0):.2f}"
            if getattr(node, "answering_confidence", None) is not None:
                label += f" a={getattr(node, 'answering_confidence', 0):.2f}"
            items.append((crop, label))
        self._save_crop_grid(filename, items, title)

    def _save_crop_grid(self, filename, items, title, max_cols=4):
        if not items:
            return
        thumbs = [(_fit_thumb(img), label) for img, label in items]
        cell_w, cell_h = 280, 220
        cols = min(max_cols, len(thumbs))
        rows = math.ceil(len(thumbs) / cols)
        canvas = Image.new("RGB", (cols * cell_w, rows * cell_h + 28), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), title[:160], fill=(0, 0, 0), font=ImageFont.load_default())
        for idx, (thumb, label) in enumerate(thumbs):
            col = idx % cols
            row = idx // cols
            x = col * cell_w + 8
            y = row * cell_h + 32
            canvas.paste(thumb, (x, y))
            draw.text((x, y + thumb.size[1] + 4), label[:60], fill=(0, 0, 0), font=ImageFont.load_default())
        canvas.save(self._path(filename))

    def record_second_crop(self, name, image_pil, crop_box, target):
        if not crop_box:
            return
        event_slug = _safe_slug(name)
        cropped = image_pil.crop(crop_box)
        cropped.save(self._path(f"05_{event_slug}_crop.png"))
        if self.original_image is not None:
            out = self.original_image.copy()
            draw = ImageDraw.Draw(out)
            _draw_box(draw, crop_box, (200, 0, 200), f"second crop: {target}", width=5)
            out.save(self._path(f"05_{event_slug}_crop_global.png"))

    def start_search(self, label, visual_cue, question, image_pil):
        self.trace_counter += 1
        trace_id = self.trace_counter
        self.traces[trace_id] = {
            "label": label or f"search_{trace_id}",
            "visual_cue": visual_cue,
            "question": question,
            "image": image_pil.copy(),
            "steps": [],
        }
        return trace_id

    def record_search_step(self, trace_id, stage_name, node, answering_confidence, threshold):
        if trace_id not in self.traces:
            return
        self.traces[trace_id]["steps"].append(
            {
                "stage": stage_name,
                "bbox_xywh": list(map(int, node.state.bbox)),
                "node_id": getattr(node, "id", None),
                "depth": getattr(node, "depth", None),
                "answering_confidence": float(answering_confidence),
                "threshold": float(threshold),
                "prior_prob": float(getattr(node, "prior_prob", 0)),
                "fast_confidence": None if getattr(node, "fast_confidence", None) is None else float(node.fast_confidence),
                "posterior_score": float(getattr(node, "posterior_score", 0)),
            }
        )

    def finish_search(self, trace_id, success, result_nodes):
        if trace_id not in self.traces:
            return
        trace = self.traces[trace_id]
        slug = _safe_slug(f"{trace_id:02d}_{trace['label']}")
        trace_json = dict(trace)
        trace_json.pop("image", None)
        trace_json["success"] = bool(success)
        trace_json["result_bboxes_xywh"] = [list(map(int, n.state.bbox)) for n in result_nodes]
        self.write_json(f"06_{slug}_trace.json", trace_json)
        self._save_search_overlay(f"06_{slug}_trace.png", trace["image"], trace["steps"], result_nodes, trace["label"])
        items = []
        for step in trace["steps"][: self.max_trace_crops]:
            box = _clip_xyxy(_xywh_to_xyxy(step["bbox_xywh"]), trace["image"])
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            crop = trace["image"].crop(box)
            label = f"{len(items) + 1} {step['stage']} d={step['depth']} a={step['answering_confidence']:.2f}"
            items.append((crop, label))
        self._save_crop_grid(f"06_{slug}_crops.png", items, f"Search trace crops: {trace['label']}")

    def _save_search_overlay(self, filename, image_pil, steps, result_nodes, title):
        out = image_pil.copy()
        draw = ImageDraw.Draw(out)
        for idx, step in enumerate(steps):
            color = (220, 80, 0)
            label = f"{idx + 1} a={step['answering_confidence']:.2f}"
            _draw_box(draw, _xywh_to_xyxy(step["bbox_xywh"]), color, label, width=2)
        for idx, node in enumerate(result_nodes):
            _draw_box(draw, _xywh_to_xyxy(node.state.bbox), (0, 180, 0), f"RESULT {idx + 1}", width=5)
        _draw_label(draw, (6, 6), title[:120], (0, 0, 0))
        out.save(self._path(filename))

    def record_final(self, annotation, image_pil, searched_nodes, response):
        out = image_pil.copy()
        draw = ImageDraw.Draw(out)
        for i, bbox in enumerate(annotation.get("bbox") or []):
            _draw_box(draw, _xywh_to_xyxy(bbox), (0, 160, 0), f"GT {i + 1}", width=4)
        for i, node in enumerate(searched_nodes):
            source = getattr(node, "search_source", "node")
            _draw_box(draw, _xywh_to_xyxy(node.state.bbox), (220, 30, 30), f"FINAL {i + 1}:{source}", width=5)
        self._draw_question_band(out, annotation)
        out.save(self._path("09_final_evidence_overlay.png"))
        items = []
        for i, node in enumerate(searched_nodes):
            box = _clip_xyxy(_xywh_to_xyxy(node.state.bbox), image_pil)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            crop = image_pil.crop(box)
            items.append((crop, f"FINAL {i + 1} {getattr(node, 'search_source', 'node')}"))
        self._save_crop_grid("09_final_evidence_crops.png", items, "Final visual evidence crops")
        self.write_json(
            "09_final_summary.json",
            {
                "response": response,
                "searched_bbox_xywh": [list(map(int, n.state.bbox)) for n in searched_nodes],
                "search_sources": [getattr(n, "search_source", "node") for n in searched_nodes],
                "annotation": annotation,
            },
        )
