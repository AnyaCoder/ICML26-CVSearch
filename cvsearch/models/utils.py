from PIL import Image, ImageDraw
import numpy as np
import os
import json
import math
import logging
import re
from copy import deepcopy
from typing import List
# For the visual cues like "man and his bag", we should remove the pronoun "his bag"
def include_pronouns(nlp, text):
    doc = nlp(text)
    for token in doc:
        if token.pos_ == 'PRON':
            return True
    return False


def extract_visual_objects(nlp, text):
    doc = nlp(text)
    objects = []

    stop_nouns = set([
        "color", "position", "size", "shape", "texture", "material", "what", "where", "kind", "type",
        "side", "corner", "part", "surface", "area", "region", "level", "spot", "direction",
        "picture", "image", "photo", "scene", "background", "view",
        "left", "right", "top", "bottom", "front", "back", "middle", "center",
        "how", "many", "brand", "name", "object", "thing", "map", "locations", "country"
    ])

    wh_words = set(["which", "what", "whose", "who", "that"])

    def clean_chunk(text_span):
        return re.sub(r'^(the|a|an|this|that|these|those)\s+', '', text_span, flags=re.IGNORECASE).strip()

    def get_smart_expanded_span(chunk):
        root = chunk.root
        min_i = chunk.start
        max_i = chunk.end - 1

        def traverse(node):
            nonlocal max_i
            for child in node.rights:
                if child.dep_ in ['compound', 'amod']:
                    if child.i > max_i:
                        max_i = child.i
                    traverse(child)

                elif child.dep_ in ['prep', 'acl', 'pobj', 'relcl']:
                    is_spatial_bridge = False
                    if child.dep_ == 'acl' and child.lemma_ in ['locate', 'position', 'situate', 'place']:
                        is_spatial_bridge = True

                    if child.dep_ == 'prep':
                        for grandchild in child.rights:
                            if grandchild.dep_ == 'pobj':
                                if grandchild.lemma_.lower() in stop_nouns:
                                    is_spatial_bridge = True
                                break

                    if not is_spatial_bridge:
                        if child.i > max_i:
                            max_i = child.i
                        traverse(child)

        traverse(root)
        final_span = doc[min_i: max_i + 1]
        return final_span.text

    candidates = []
    for chunk in doc.noun_chunks:
        root = chunk.root
        if chunk[0].text.lower() in wh_words:
            continue

        if root.pos_ == 'PRON' or root.lemma_.lower() in stop_nouns:
            continue

        condition = (
                root.dep_ in ["dobj", "nsubj", "nsubjpass", "ROOT", "attr", "pobj", "conj", "appos"]
        )

        if condition:
            expanded_text = get_smart_expanded_span(chunk)
            clean_text = clean_chunk(expanded_text)
            if clean_text.lower() not in stop_nouns and clean_text:
                candidates.append({
                    "text": clean_text,
                    "root_idx": root.i,
                    "length": len(clean_text)
                })

    final_candidates = []
    candidates.sort(key=lambda x: x['length'], reverse=True)

    for cand in candidates:
        current_text = cand['text']
        is_contained = False

        for kept in final_candidates:
            if current_text in kept['text'] and current_text != kept['text']:
                is_contained = True
                break

        if not is_contained:
            final_candidates.append(cand)

    final_candidates.sort(key=lambda x: x['root_idx'])

    objects = [c['text'] for c in final_candidates]
    if not objects:
        return [text.strip()]

    if len(objects) > 5:
        objects = objects[:5]

    return objects


def extract_targets(sentence: str, pattern=r"So I need the information about the following objects: (.+)"):
    match = re.search(pattern, sentence)
    if match:
        return match.group(1)
    return None


def extract_targets_SGVS(sentence: str, pattern=r":\s*(.+)$"):
    match = re.search(pattern, sentence)
    if match:
        return match.group(1)
    return None


def split_targets_sentence(targets_sentence: str, split_tag=r' and |, '):
    if targets_sentence.endswith('.'):
        targets_sentence = targets_sentence[:-1]
    targets = re.split(split_tag, targets_sentence)
    return targets


def extract_targets_optimized(sentence: str):
    if not sentence:
        return None

    pattern = r"(?:objects|targets|items)(?:\s*include)?\s*[:：]\s*(.+)$"
    match = re.search(pattern, sentence, re.IGNORECASE | re.DOTALL)

    if match:
        targets_str = match.group(1).strip()
    else:
        fallback_match = re.search(r"[:：]\s*(.+)$", sentence, re.DOTALL)
        if fallback_match:
            targets_str = fallback_match.group(1).strip()
        else:
            return None

    if targets_str.endswith('.'):
        targets_str = targets_str[:-1]

    return targets_str

def parse_targets_str(targets_str):
    if isinstance(targets_str, list):
        return targets_str
    if not targets_str or not isinstance(targets_str, str):
        return []
    # -----------------------
    split_pattern = r',\s*(?:and|or)\s+|\s+(?:and|or)\s+|[,;]\s+'

    raw_targets = re.split(split_pattern, targets_str)

    clean_targets = []
    for t in raw_targets:
        t = t.strip()
        t = t.strip('\'"')

        if t and t.lower() != 'none' and t.lower() != 'null':
            clean_targets.append(t)

    return clean_targets

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return deepcopy(pil_img), 0, 0
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result, 0, (width - height) // 2
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result, (height - width) // 2, 0

def bbox_area(bbox):
    x_min, y_min, x_max, y_max = bbox
    return (x_max - x_min) * (y_max - y_min)

def intersect_bbox(bboxA, bboxB, distance_buffer=50):
    bbox1 = [v - distance_buffer if i < 2 else v + distance_buffer for i, v in enumerate(bboxA)]
    bbox2 = [v - distance_buffer if i < 2 else v + distance_buffer for i, v in enumerate(bboxB)]
    """ Calculate the union of two bounding boxes. """
    x_min = max(bbox1[0], bbox2[0])
    y_min = max(bbox1[1], bbox2[1])
    x_max = min(bbox1[2], bbox2[2])
    y_max = min(bbox1[3], bbox2[3])

    if x_max > x_min and y_max > y_min:
        return (x_min, y_min, x_max, y_max)

    return None

def merge_bboxes(bbox1, bbox2):
    return (
        min(bbox1[0], bbox2[0]),
        min(bbox1[1], bbox2[1]),
        max(bbox1[2], bbox2[2]),
        max(bbox1[3], bbox2[3])
    )

def merge_bbox_list(bboxes, threshold=0):
    """merge all cross bboxes in the List bboxes"""
    changed = True
    while changed:
        changed = False
        new_bboxes = []
        used = set()

        for i in range(len(bboxes)):
            if i in used:
                continue
            merged = False

            for j in range(len(bboxes)):
                if j in used or i == j:
                    continue
                intersection = intersect_bbox(bboxes[i], bboxes[j])
                if intersection:
                    if threshold == 0 or (threshold > 0 and (
                            bbox_area(intersection) >= threshold * bbox_area(bboxes[i]) or bbox_area(
                            intersection) >= threshold * bbox_area(bboxes[j]))):
                        new_bbox = merge_bboxes(bboxes[i], bboxes[j])
                        new_bboxes.append(new_bbox)
                        used.update([i, j])
                        changed = True
                        merged = True
                        break
            if not merged and i not in used:
                new_bboxes.append(bboxes[i])

        bboxes = new_bboxes

    return bboxes


def merge_bbox_list_sgavs(bboxes, threshold=0):
    """
    merge all cross bboxes in the List bboxes
    Modification:
    1. Original merge logic kept intact
    2. Save contained bbox before merge
    3. Append saved bboxes to final result
    """
    saved_contained_bboxes = []
    changed = True
    while changed:
        changed = False
        new_bboxes = []
        used = set()

        for i in range(len(bboxes)):
            if i in used:
                continue
            merged = False

            for j in range(len(bboxes)):
                if j in used or i == j:
                    continue

                intersection = intersect_bbox(bboxes[i], bboxes[j])
                if intersection:
                    if threshold == 0 or (threshold > 0 and (
                            bbox_area(intersection) >= threshold * bbox_area(bboxes[i]) or bbox_area(
                        intersection) >= threshold * bbox_area(bboxes[j]))):

                        area_i = bbox_area(bboxes[i])
                        area_j = bbox_area(bboxes[j])
                        area_inter = bbox_area(intersection)

                        if area_inter >= area_i and area_i < area_j:
                            saved_contained_bboxes.append(bboxes[i])

                        elif area_inter >= area_j and area_j < area_i:
                            saved_contained_bboxes.append(bboxes[j])
                        # ---------------------------------------------
                        new_bbox = merge_bboxes(bboxes[i], bboxes[j])
                        new_bboxes.append(new_bbox)
                        used.update([i, j])
                        changed = True
                        merged = True
                        break

            if not merged and i not in used:
                new_bboxes.append(bboxes[i])

        bboxes = new_bboxes

    return bboxes + saved_contained_bboxes


def union_all_bboxes(bboxes):
    if len(bboxes) == 0:
        return None
    ret = bboxes[0]
    for bbox in bboxes[1:]:
        ret = merge_bboxes(ret, bbox)
    return ret


def union_blocks_independent(full_image: Image.Image, bbox1, bbox2, resized_long, backgroud_color):
    if bbox1[0] > bbox2[0]:
        bbox1, bbox2 = bbox2, bbox1
    block1 = full_image.crop(bbox1).resize((resized_long, resized_long))
    block2 = full_image.crop(bbox2).resize((resized_long, resized_long))
    background = Image.new('RGB', (2 * resized_long, 2 * resized_long), backgroud_color)
    center_y1 = (bbox1[3] + bbox1[1]) // 2
    center_y2 = (bbox2[3] + bbox2[1]) // 2
    offset_y = center_y2 - center_y1
    offset_y = np.clip(offset_y, -resized_long, resized_long)
    paste_x1 = 0
    paste_y1 = resized_long // 2 - offset_y // 2
    paste_x2 = resized_long
    paste_y2 = resized_long // 2 + offset_y // 2

    background.paste(block1, (paste_x1, paste_y1))
    background.paste(block2, (paste_x2, paste_y2))

    return background


def visualize_bbox_and_arrow(image: Image.Image, bbox, color="red", thickness=2, xyxy=False):
    """Visualizes a single bounding box on the image"""
    if not xyxy:
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
    else:
        x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - thickness)
    y1 = max(0, y1 - thickness)
    x2 = min(image.width, x2 + thickness)
    y2 = min(image.height, y2 + thickness)
    draw = ImageDraw.Draw(image)
    new_bbox = [x1, y1, x2, y2]
    draw.rectangle((x1, y1, x2, y2), outline=color, width=thickness)
    min_distance = thickness * 6
    center_x = image.width // 2
    center_y = image.height // 2
    center_x_bbox = (x1 + x2) // 2
    center_y_bbox = (y1 + y2) // 2
    return new_bbox


def normalize_target_text(t_target):
    is_type2 = False

    if t_target:
        t_target = re.sub(r'^[\W_]+|[\W_]+$', '', t_target)
        if t_target.startswith("all "):
            is_type2 = True
            t_target = t_target[4:]
            if t_target.endswith('s'):
                t_target = t_target[:-1]

    return t_target, is_type2

def should_merge(elem1, elem2, n):
    box1 = elem1["bounding_box_added"]
    box2 = elem2["bounding_box_added"]
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    if x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1:
        dx = max(0, x1_1 - x2_2, x1_2 - x2_1)
        dy = max(0, y1_1 - y2_2, y1_2 - y2_1)
        distance = math.sqrt(dx ** 2 + dy ** 2)
        return distance < n
    else:
        return True

def merge_two_elements(elem1, elem2):
    box1 = elem1["bounding_box_added"]
    box2 = elem2["bounding_box_added"]

    merged_element = {
        "id": min(elem1["id"], elem2["id"]),
        "area": elem1["area"] + elem2["area"],
        "bounding_box_added": [
            min(box1[0], box2[0]),
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3])
        ]
    }
    return merged_element

def merge_elements_in_category(elements, n):
    if len(elements) < 2:
        return elements
    while True:
        merged_in_this_pass = False
        i = 0
        while i < len(elements):
            j = i + 1
            while j < len(elements):
                if should_merge(elements[i], elements[j], n):
                    new_element = merge_two_elements(elements[i], elements[j])
                    elements.pop(j)
                    elements.pop(i)
                    elements.append(new_element)
                    merged_in_this_pass = True
                    break
                else:
                    j += 1
            if merged_in_this_pass:
                break
            else:
                i += 1
        if not merged_in_this_pass:
            break
    return elements

def process_and_merge_data(data):
    category_bboxes = {}

    for item in data:
        category_name = item["category_name"]
        bbox = item["bounding_box_added"]
        if category_name not in category_bboxes:
            category_bboxes[category_name] = []
        category_bboxes[category_name].append(bbox)

    def union_bbox(bboxes):
        x1 = min(b[0] for b in bboxes)
        y1 = min(b[1] for b in bboxes)
        x2 = max(b[2] for b in bboxes)
        y2 = max(b[3] for b in bboxes)
        return [x1, y1, x2, y2]

    flat_format = []
    for category_name, bboxes in category_bboxes.items():
        merged_bbox = union_bbox(bboxes)
        flat_format.append({
            'object': str(category_name),
            'bbox': merged_bbox
        })

    return flat_format


def join_semantic_list(ranked_semantic_list):
    return ', '.join(str(item) for item in ranked_semantic_list)


def extract_semantic_list(semantic_list_flat):
    semantic_name_list = []
    unknown_list = []
    for semantic in semantic_list_flat:
        semantic_name_list.append(semantic.get("object"))
        if semantic.get("object") == "unknown":
            unknown_list.append(semantic.get("object"))

    return semantic_name_list, unknown_list


def load_json_or_jsonl(data_path):
    _, ext = os.path.splitext(data_path)
    with open(data_path, 'r', encoding='utf-8') as f:
        if ext.lower() == '.json':
            return json.load(f)
        elif ext.lower() == '.jsonl':
            return [json.loads(line.strip()) for line in f if line.strip()]
        else:
            raise ValueError(f"Unsupported file format: {ext}")


def filter_ranked_by_semantic_whitelist(semantic_name_list, ranked_semantic_list):
    """
    Filter the MLLM's ranked list to only include items present in the original semantic list,
    preserving the order from the ranked list and removing duplicates.

    Args:
        semantic_name_list (list of str): Unordered whitelist of valid regions.
        ranked_semantic_list (list of str or str): MLLM output (ordered, may have extras/dups).

    Returns:
        dict: {
            "filtered_ranked": list[str],      # Valid, deduplicated, ordered list
            "missing_from_ranked": list[str],  # Items in semantic but not in ranked
            "invalid_in_ranked": list[str],    # Items in ranked but not in semantic
            "duplicates_removed": list[str]    # Duplicate items removed (after first occurrence)
        }
    """
    # Normalize inputs
    if isinstance(ranked_semantic_list, str):
        ranked = [item.strip() for item in ranked_semantic_list.split(',')]
    else:
        ranked = list(ranked_semantic_list)

    semantic_set = set(semantic_name_list)
    semantic_list = list(semantic_name_list)  # keep for missing check

    filtered = []
    seen = set()
    invalid_items = []
    duplicate_items = []

    for item in ranked:
        if item not in semantic_set:
            invalid_items.append(item)
        elif item in seen:
            duplicate_items.append(item)
        else:
            filtered.append(item)
            seen.add(item)

    # Items in semantic but never appeared in ranked (even after filtering)
    missing = [item for item in semantic_list if item not in seen]

    return {
        "filtered_ranked": filtered,
        "missing_from_ranked": missing,
        "invalid_in_ranked": invalid_items,
        "duplicates_removed": duplicate_items
    }


def get_bbox_by_object(data, obj_name):
    for item in data:
        if item['object'] == obj_name:
            return item['bbox']
    return None

def compute_iou(box1: List[int], box2: List[int]) -> float:
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


class RegionTracker:
    def __init__(self, iou_threshold: float = 0.5):
        self.searched_regions: List[dict] = []
        self.iou_threshold = iou_threshold

    def is_already_searched(self, new_box: List[int]) -> bool:
        x1, y1, x2, y2 = new_box
        new_area = (x2 - x1) * (y2 - y1)
        if new_area == 0:
            return False

        for record in self.searched_regions:
            searched_box = record['box']
            iou_val = compute_iou(new_box, searched_box)
            if iou_val >= self.iou_threshold:
                return True

            inter_area = self._compute_intersection_area(new_box, searched_box)
            if inter_area > 0:
                coverage_ratio = inter_area / new_area
                if coverage_ratio >= self.iou_threshold:
                    return True

        return False

    def _compute_intersection_area(self, box1: List[int], box2: List[int]) -> float:
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2

        inter_x1 = max(x1, x1_p)
        inter_y1 = max(y1, y1_p)
        inter_x2 = min(x2, x2_p)
        inter_y2 = min(y2, y2_p)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        return (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    def should_search(self, new_box: List[int]) -> bool:
        return not self.is_already_searched(new_box)

    def add_searched_region(self, box: List[int], confidence: float):
        self.searched_regions.append({
            'box': box,
            'confidence': confidence
        })

    def get_searched_regions(self) -> List[dict]:
        return self.searched_regions.copy()

    def clear(self):
        self.searched_regions.clear()


def parse_output(output_text, w_v=0.5, w_c=0.5):
    regions = {}
    valid_lines = 0

    if not (0 <= w_v <= 1 and 0 <= w_c <= 1 and abs(w_v + w_c - 1.0) < 1e-5):
        raise ValueError("The weight must meet: 0<=w_v<=1, 0<=w_c<=1, w_v + w_c = 1")

    pattern = r'([^|]+)\s*\|\s*v=([0-9.]+)\s*\|\s*c=([0-9.]+)'
    valid_regions = []
    for line in output_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.search(pattern, line)
        if match:
            region_name = match.group(1).strip()
            v = float(match.group(2))
            c = float(match.group(3))
            if not (0.0 <= v <= 1.0 and 0.0 <= c <= 1.0):
                logging.warning(f"Score exceeds the range: {region_name} | v={v}, c={c} ([0.0,1.0])")
                v = max(0.0, min(1.0, v))
                c = max(0.0, min(1.0, c))

            valid_regions.append((region_name, v, c))
            valid_lines += 1

    if valid_regions:
        v_avg = sum(v for _, v, _ in valid_regions) / len(valid_regions)
        c_avg = sum(c for _, _, c in valid_regions) / len(valid_regions)
        all_invalid = False
    else:
        v_avg = 0.0
        c_avg = 0.0
        all_invalid = True

    for region_name, v, c in valid_regions:
        total_score = w_v * v + w_c * c
        regions[region_name] = {
            'visual_score': v,
            'commonsense_score': c,
            'total_score': total_score
        }

    ranked_list = [region for region, _ in sorted(regions.items(), key=lambda x: x[1]['total_score'], reverse=True)]
    t_avg = w_v * v_avg + w_c * c_avg
    return {
        'regions': regions,
        'ranked_list': ranked_list,
        'v_avg': v_avg,
        'c_avg': c_avg,
        't_avg': t_avg,
        'all_invalid': all_invalid
    }
