from models.tree import ImageTree, Node, NodeState, AdaptiveImageTree, NodeA
from models.utils import include_pronouns, load_json_or_jsonl, extract_visual_objects, normalize_target_text
from models.modeling_sam3 import ConstrainedTreeBuilder
from typing import Callable, List, Tuple, Union
from PIL import Image
from copy import deepcopy
import os
import numpy as np
import torch

def get_cvsearch_response(
        sam_model,
        zoom_model,
        nlp_model,
        annotation,
        ic_examples,
        decomposed_question_template,
        answering_confidence_threshold_upper,
        answering_confidence_threshold_lower,
        fast_threshold,
        pop_limit,
        threshold_descrease,
        image_folder: str = None,
        search_mode=True,
        enable_parent_verification=False,
        debug_recorder=None,
        evidence_compiler=None,
):
    # Data loading
    #Default single_target: tree_depth_s = 2, cross_target: tree_depth_c = 3
    keep_threshold_mllm_primary = 0.15
    keep_threshold_rule_primary = 0.25
    keep_threshold_second = 0.15
    tree_prune_threshold = 0.4
    tree_depth_s = 2
    tree_depth_c = 3
    input_image = annotation['input_image']
    if image_folder is not None:
        input_image = os.path.join(image_folder, input_image)
    image_pil = Image.open(input_image).convert('RGB')
    if debug_recorder is not None:
        debug_recorder.start_sample(annotation, image_pil, input_image)
    question = annotation['question']
    options = annotation.get('options', None)
    question_free_form = None
    searched_nodes = []
    ####Quick assessment
    img_w, img_h = image_pil.size
    state = NodeState(image_pil, [0, 0, img_w, img_h])
    root_node = NodeA(state)
    root_node.is_root = True
    root_node.search_source = "global"
    root_ans_conf = zoom_model.get_confidence_value([root_node], image_pil, confidence_type='answering',input_ele=question)
    annotation['root_ans_conf'] = root_ans_conf
    if debug_recorder is not None:
        debug_recorder.record_root_confidence(root_ans_conf)
    annotation['sam'] = []
    if root_ans_conf>answering_confidence_threshold_lower+fast_threshold:
        #####Quick Answer########
        # print("Quick Answer!")
        searched_nodes.append(root_node)
        annotation['targets'] = None
        annotation['target_sign'] = None
        annotation['num_pop'] = []
        annotation['num_zoom_in'] = []
        annotation['num_zoom_out'] = []
        annotation['search_mode'] = 0
        annotation['sam'].append(None)
    else:
        #####Visual Search########
        # print("Visual Search!")
        # key object extract
        targets = zoom_model.generate_visual_cues_using_ic(ic_examples, question)
        #For the visual cues like "man and his bag", we should remove the pronoun "his bag"
        targets = [t for t in targets if not include_pronouns(nlp_model, t)]
        #For the visual cues like "all dogs", we convert it to "dog"
        processed_results = [normalize_target_text(t) for t in targets]
        targets = [res[0] for res in processed_results]
        # is_type2_triggered = any(res[1] for res in processed_results)
        is_search_second =False
        # sam3
        target_sign = True if len(targets)>0 else False
        ###MLLM extract key objects
        if target_sign:
            text_target = targets
            with torch.inference_mode():
                backbone_out, processed_results, target_id = sam_model.batch_inference(image_pil, text_target)
        else:
            text_target = extract_visual_objects(nlp_model, question)
            with torch.inference_mode():
                backbone_out, processed_results, target_id = sam_model.batch_inference(image_pil, text_target)
        one_target_search = (len(text_target) == 1)

        # print("targets:", targets)
        # print('text_target:', text_target)
        annotation['targets'] = text_target
        annotation['target_sign'] = target_sign
        annotation['num_pop'] = []
        annotation['num_zoom_in'] = []
        annotation['num_zoom_out'] = []

        ####sam3 result -> bbox
        sam_success_flags, sam_bboxes = process_sam_result(processed_results, target_id, is_search_second)
        if debug_recorder is not None:
            debug_recorder.record_sam(
                "primary_sam",
                image_pil,
                text_target,
                processed_results,
                target_id,
                sam_success_flags,
                sam_bboxes,
            )

        # Adaptive visual search
        if target_sign:
            # print('MLLM Visual Cue!')
            # MLLM extract target objects
            if sum(sam_success_flags) == len(text_target):
                # sam3 segment all target objects
                fast_node = []
                for search_box, t_target in zip(sam_bboxes, text_target):
                    x0, y0 = search_box[0], search_box[1]
                    w, h = search_box[2] - search_box[0], search_box[3] - search_box[1]
                    bbox_xywh = [x0, y0, w, h]
                    state = NodeState(image_pil, bbox_xywh)
                    node = NodeA(state)
                    node.search_source = "fast"
                    node.target_phrase = t_target
                    fast_node.append(node)

                # print("Fast Search Success!")
                searched_nodes.extend(fast_node)
                num_pop = 0
                num_zoom_in = 0
                num_zoom_out = 0
                annotation['num_pop'].append(num_pop)
                annotation['num_zoom_in'].append(num_zoom_in)
                annotation['num_zoom_out'].append(num_zoom_out)
                annotation['search_mode'] = 1
                annotation['sam'].append(True)
            #Fast search fail
            else:
                # print("Fast Search Fail!")
                zoom_node = [] #
                #sam3 segment partial target objects
                image_features_batch = backbone_out['vision_features']
                if isinstance(image_features_batch, torch.Tensor):
                    feat = image_features_batch.detach().cpu().float().numpy()
                else:
                    feat = image_features_batch

                del backbone_out
                del image_features_batch

                feat = feat.squeeze(0)  # batch -> (256, 72, 72), C,H,W
                tree_depth = 3
                builder = ConstrainedTreeBuilder(feat, n_atoms=600, pos_weight=3.5, split_threshold=0.3, keep_threshold=keep_threshold_mllm_primary)
                tree_dict = builder.build_tree(max_depth=tree_depth, min_splits=4, max_splits=8)
                feat_shape = feat.shape
                image_tree = AdaptiveImageTree(image_pil, tree_dict, feat_shape)
                if debug_recorder is not None:
                    debug_recorder.record_tree("primary_tree", image_tree)
                    debug_recorder.record_tree_boundaries("primary_tree", image_pil, builder, tree_dict)
                num_pop = []
                for flag, search_box, t_target in zip(sam_success_flags, sam_bboxes, text_target):
                    if flag==1:
                        #Successfully Segment
                        x0, y0 = search_box[0], search_box[1]
                        w, h = search_box[2] - search_box[0], search_box[3] - search_box[1]
                        bbox_xywh = [x0, y0, w, h]
                        state = NodeState(image_pil, bbox_xywh)
                        node = NodeA(state)
                        node.search_source = "fast"
                        node.target_phrase = t_target
                        zoom_node.append(node)
                        num_pop.append(1)
                        annotation['sam'].append(True)
                    else:
                        annotation['sam'].append(False)
                        candidates_search, num_pop_search, is_success = semantic_guide_search_dynamic_depth(
                            zoom_model=zoom_model,
                            pop_limit=pop_limit,
                            num_intervel=2,
                            threshold_descrease=threshold_descrease,
                            depth_limit=tree_depth_s if one_target_search else tree_depth_c,
                            question=question if one_target_search else decomposed_question_template.format(t_target),
                            visual_cue=t_target,
                            answering_confidence_threshold_lower=answering_confidence_threshold_lower,
                            answering_confidence_threshold_upper=answering_confidence_threshold_upper,
                            image_pil=image_pil,
                            image_tree=image_tree,
                            enable_parent_verification=enable_parent_verification,
                            prior_pruning_threshold=tree_prune_threshold,
                            debug_recorder=debug_recorder,
                            debug_label=f"primary_{t_target}",
                        )
                        num_pop.append(num_pop_search)
                        if is_success:
                            #Search successfully
                            for cand in candidates_search:
                                cand.search_source = "fine"
                                cand.target_phrase = t_target

                            zoom_node.extend(candidates_search)
                            # print("Fine Search Success!")
                        else:
                            # Search fail: candidates_search is the sorted node list
                            if candidates_search:
                                best_candidate = candidates_search[0]  # first Node
                                if not search_mode:
                                    best_candidate.search_source = "fine_fallback"
                                    best_candidate.target_phrase = t_target
                                    zoom_node.append(best_candidate)
                                    # print("Fine Search Fail!")
                                else:
                                    ###Second search
                                    is_search_second = True
                                    cropped_image, cropped_bbox = crop_image_by_node(image_pil, best_candidate)
                                    if cropped_image:
                                        left, top = cropped_bbox[0], cropped_bbox[1]
                                        if debug_recorder is not None:
                                            debug_recorder.record_second_crop(f"second_{t_target}", image_pil, cropped_bbox, t_target)
                                        with torch.inference_mode():
                                            backbone_out_sub, processed_results_sub, target_id_sub = sam_model.batch_inference(cropped_image, [t_target])
                                        image_features_batch_sub = backbone_out_sub['vision_features']
                                        if isinstance(image_features_batch_sub, torch.Tensor):
                                            feat_sub = image_features_batch_sub.detach().cpu().float().numpy()
                                        else:
                                            feat_sub = image_features_batch_sub
                                        feat_sub = feat_sub.squeeze(0)
                                        del backbone_out_sub
                                        del image_features_batch_sub

                                        sam_success_flags_sub, sam_bboxes_sub = process_sam_result(processed_results_sub, target_id_sub, is_search_second)
                                        if debug_recorder is not None:
                                            debug_recorder.record_sam(
                                                f"second_sam_{t_target}",
                                                cropped_image,
                                                [t_target],
                                                processed_results_sub,
                                                target_id_sub,
                                                sam_success_flags_sub,
                                                sam_bboxes_sub,
                                                offset=(left, top),
                                            )
                                        if sum(sam_success_flags_sub) == len([t_target]):
                                            # second sam3 inference successfully segmented target objects
                                            fast_node = []
                                            for search_box in sam_bboxes_sub:
                                                x0, y0 = search_box[0], search_box[1]
                                                w, h = search_box[2] - search_box[0], search_box[3] - search_box[1]
                                                bbox_xywh = [x0+left, y0+top, w, h]  ###bbox offset
                                                state = NodeState(image_pil, bbox_xywh)
                                                node = NodeA(state)
                                                node.search_source = "fast"
                                                node.target_phrase = t_target
                                                fast_node.append(node)

                                            # print("Second Fast Search Success!")
                                            zoom_node.extend(fast_node)
                                        else:
                                            # second segmentation still failed
                                            tree_depth_sub = tree_depth_s
                                            depth_limit_sub = tree_depth_s
                                            builder_sub = ConstrainedTreeBuilder(feature_map=feat_sub, n_atoms=600,
                                                                                 pos_weight=3.5, split_threshold=0.3,
                                                                                 keep_threshold=keep_threshold_second,
                                                                                 use_local_normalization=True,
                                                                                 use_silhouette_score=True)
                                            tree_sub = builder_sub.build_tree(max_depth=tree_depth_sub, min_splits=4, max_splits=8)
                                            feat_shape_sub = feat_sub.shape
                                            image_tree_sub = AdaptiveImageTree(cropped_image, tree_sub, feat_shape_sub)
                                            if debug_recorder is not None:
                                                debug_recorder.record_tree(f"second_tree_{t_target}", image_tree_sub)
                                                debug_recorder.record_tree_boundaries(f"second_tree_{t_target}", cropped_image, builder_sub, tree_sub)
                                            candidates_search_sub, num_pop_search_sub, is_success_sub = semantic_guide_search_dynamic_depth(
                                                zoom_model=zoom_model,
                                                pop_limit=pop_limit,
                                                num_intervel=2,
                                                threshold_descrease=threshold_descrease,
                                                depth_limit=depth_limit_sub,
                                                question=question if one_target_search else decomposed_question_template.format(t_target),
                                                visual_cue=t_target,
                                                answering_confidence_threshold_lower=answering_confidence_threshold_lower,
                                                answering_confidence_threshold_upper=answering_confidence_threshold_upper,
                                                image_pil=cropped_image,
                                                image_tree=image_tree_sub,
                                                enable_parent_verification=enable_parent_verification,
                                                prior_pruning_threshold=tree_prune_threshold,
                                                debug_recorder=debug_recorder,
                                                debug_label=f"second_{t_target}",
                                            )

                                            if is_success_sub:
                                                second_search_node=candidates_search_sub[0]
                                                bbox = second_search_node.state.bbox
                                                x0, y0, w, h = bbox
                                                bbox_shifted = [x0+left, y0+top, w, h]
                                                state = NodeState(image_pil, bbox_shifted)
                                                node = NodeA(state)
                                                node.search_source = "fine"
                                                node.target_phrase = t_target
                                                zoom_node.append(node)
                                                # print("Second Fine Search Success!")
                                            else:
                                                best_candidate.search_source = "fine_fallback"
                                                best_candidate.target_phrase = t_target
                                                zoom_node.append(best_candidate)
                                                # print("Fine Search Fail!")

                if len(zoom_node) > 0:
                    searched_nodes.extend(zoom_node)
                    num_zoom_in = 0
                    num_zoom_out = 0
                    annotation['num_pop'].append(num_pop)
                    annotation['num_zoom_in'].append(num_zoom_in)
                    annotation['num_zoom_out'].append(num_zoom_out)
                    annotation['search_mode'] = 2
                else:
                    annotation['search_mode'] = 3

        else:
            # print('Rules Visual Cue!')
            # MLLM extraction failed, rule matches target objects
            if sum(sam_success_flags) == len(text_target):
                fast_node = []
                for search_box, t_target in zip(sam_bboxes, text_target):
                    x0, y0 = search_box[0], search_box[1]
                    w, h = search_box[2] - search_box[0], search_box[3] - search_box[1]
                    bbox_xywh = [x0, y0, w, h]
                    state = NodeState(image_pil, bbox_xywh)
                    node = NodeA(state)
                    node.search_source = "fast"
                    node.target_phrase = t_target
                    fast_node.append(node)

                # print("Fast Search Success!")
                searched_nodes.extend(fast_node)
                num_pop = 0
                num_zoom_in = 0
                num_zoom_out = 0
                annotation['num_pop'].append(num_pop)
                annotation['num_zoom_in'].append(num_zoom_in)
                annotation['num_zoom_out'].append(num_zoom_out)
                annotation['search_mode'] = 1
                annotation['sam'].append(True)
            #Fast search fail
            else:
                # print("Fast Search Fail!")
                zoom_node = []
                image_features_batch = backbone_out['vision_features']
                if isinstance(image_features_batch, torch.Tensor):
                    feat = image_features_batch.detach().cpu().float().numpy()
                else:
                    feat = image_features_batch

                del backbone_out
                del image_features_batch

                feat = feat.squeeze(0)
                tree_depth = 3
                builder = ConstrainedTreeBuilder(feat, n_atoms=600, pos_weight=3.5, split_threshold=0.3, keep_threshold=keep_threshold_rule_primary)
                tree_dict = builder.build_tree(max_depth=tree_depth, min_splits=4, max_splits=8)
                feat_shape = feat.shape
                image_tree = AdaptiveImageTree(image_pil, tree_dict, feat_shape)
                if debug_recorder is not None:
                    debug_recorder.record_tree("primary_tree", image_tree)
                    debug_recorder.record_tree_boundaries("primary_tree", image_pil, builder, tree_dict)
                num_pop = []
                for flag, search_box, t_target in zip(sam_success_flags, sam_bboxes, text_target):
                    if flag==1:
                        x0, y0 = search_box[0], search_box[1]
                        w, h = search_box[2] - search_box[0], search_box[3] - search_box[1]
                        bbox_xywh = [x0, y0, w, h]
                        state = NodeState(image_pil, bbox_xywh)
                        node = NodeA(state)
                        node.search_source = "fast"
                        node.target_phrase = t_target
                        zoom_node.append(node)
                        num_pop.append(1)
                        annotation['sam'].append(True)
                    else:
                        annotation['sam'].append(False)
                        candidates_search, num_pop_search, is_success = semantic_guide_search_dynamic_depth(
                            zoom_model=zoom_model,
                            pop_limit=pop_limit,
                            num_intervel=2,
                            threshold_descrease=threshold_descrease,
                            depth_limit=tree_depth_s if one_target_search else tree_depth_c,
                            question=question if one_target_search else decomposed_question_template.format(t_target),
                            visual_cue=t_target,
                            answering_confidence_threshold_lower=answering_confidence_threshold_lower,
                            answering_confidence_threshold_upper=answering_confidence_threshold_upper,
                            image_pil=image_pil,
                            image_tree=image_tree,
                            enable_parent_verification=enable_parent_verification,
                            prior_pruning_threshold=tree_prune_threshold,
                            debug_recorder=debug_recorder,
                            debug_label=f"primary_{t_target}",
                        )
                        num_pop.append(num_pop_search)
                        if is_success:
                            for cand in candidates_search:
                                cand.search_source = "fine"
                            zoom_node.extend(candidates_search)
                        else:
                            if candidates_search:
                                best_candidate = candidates_search[0]
                                if not search_mode:
                                    best_candidate.search_source = "fine_fallback"
                                    best_candidate.target_phrase = t_target
                                    zoom_node.append(best_candidate)
                                    # print("Fine Search Fail!")
                                else:
                                    is_search_second = True
                                    cropped_image, cropped_bbox = crop_image_by_node(image_pil, best_candidate)
                                    if cropped_image:
                                        left, top = cropped_bbox[0], cropped_bbox[1]
                                        if debug_recorder is not None:
                                            debug_recorder.record_second_crop(f"second_{t_target}", image_pil, cropped_bbox, t_target)
                                        with torch.inference_mode():
                                            backbone_out_sub, processed_results_sub, target_id_sub = sam_model.batch_inference(
                                                cropped_image, [t_target])
                                        image_features_batch_sub = backbone_out_sub['vision_features']
                                        if isinstance(image_features_batch_sub, torch.Tensor):
                                            feat_sub = image_features_batch_sub.detach().cpu().float().numpy()
                                        else:
                                            feat_sub = image_features_batch_sub
                                        feat_sub = feat_sub.squeeze(0)
                                        del backbone_out_sub
                                        del image_features_batch_sub

                                        sam_success_flags_sub, sam_bboxes_sub = process_sam_result(
                                            processed_results_sub, target_id_sub, is_search_second)
                                        if debug_recorder is not None:
                                            debug_recorder.record_sam(
                                                f"second_sam_{t_target}",
                                                cropped_image,
                                                [t_target],
                                                processed_results_sub,
                                                target_id_sub,
                                                sam_success_flags_sub,
                                                sam_bboxes_sub,
                                                offset=(left, top),
                                            )
                                        if sum(sam_success_flags_sub) == len([t_target]):
                                            fast_node = []
                                            for search_box in sam_bboxes_sub:
                                                x0, y0 = search_box[0], search_box[1]
                                                w, h = search_box[2] - search_box[0], search_box[3] - search_box[1]
                                                bbox_xywh = [x0 + left, y0 + top, w, h]
                                                state = NodeState(image_pil, bbox_xywh)
                                                node = NodeA(state)
                                                node.search_source = "fast"
                                                node.target_phrase = t_target
                                                fast_node.append(node)

                                            # print("Second Fast Search Success!")
                                            zoom_node.extend(fast_node)
                                        else:
                                            tree_depth_sub = tree_depth_s
                                            depth_limit_sub = tree_depth_s
                                            builder_sub = ConstrainedTreeBuilder(feature_map=feat_sub, n_atoms=600,
                                                                                 pos_weight=3.5,
                                                                                 split_threshold=0.3,
                                                                                 keep_threshold=keep_threshold_second,
                                                                                 use_local_normalization=True,
                                                                                 use_silhouette_score=True)
                                            tree_sub = builder_sub.build_tree(max_depth=tree_depth_sub,
                                                                              min_splits=4, max_splits=8)
                                            feat_shape_sub = feat_sub.shape
                                            image_tree_sub = AdaptiveImageTree(cropped_image, tree_sub, feat_shape_sub)
                                            if debug_recorder is not None:
                                                debug_recorder.record_tree(f"second_tree_{t_target}", image_tree_sub)
                                                debug_recorder.record_tree_boundaries(f"second_tree_{t_target}", cropped_image, builder_sub, tree_sub)
                                            candidates_search_sub, num_pop_search_sub, is_success_sub = semantic_guide_search_dynamic_depth(
                                                zoom_model=zoom_model,
                                                pop_limit=pop_limit,
                                                num_intervel=2,
                                                threshold_descrease=threshold_descrease,
                                                depth_limit=depth_limit_sub,
                                                question=question if one_target_search else decomposed_question_template.format(t_target),
                                                visual_cue=t_target,
                                                answering_confidence_threshold_lower=answering_confidence_threshold_lower,
                                                answering_confidence_threshold_upper=answering_confidence_threshold_upper,
                                                image_pil=cropped_image,
                                                image_tree=image_tree_sub,
                                                enable_parent_verification=enable_parent_verification,
                                                prior_pruning_threshold=tree_prune_threshold,
                                                debug_recorder=debug_recorder,
                                                debug_label=f"second_{t_target}",
                                            )

                                            if is_success_sub:
                                                second_search_node = candidates_search_sub[0]
                                                bbox = second_search_node.state.bbox
                                                x0, y0, w, h = bbox
                                                bbox_shifted = [x0 + left, y0 + top, w, h]
                                                state = NodeState(image_pil, bbox_shifted)
                                                node = NodeA(state)
                                                node.search_source = "fine"
                                                node.target_phrase = t_target
                                                zoom_node.append(node)
                                                # print("Second Fine Search Success!")
                                            else:
                                                best_candidate.search_source = "fine_fallback"
                                                best_candidate.target_phrase = t_target
                                                zoom_node.append(best_candidate)
                                                # print("Fine Search Fail!")

                if len(zoom_node) > 0:
                    searched_nodes.extend(zoom_node)
                    num_zoom_in = 0
                    num_zoom_out = 0
                    annotation['num_pop'].append(num_pop)
                    annotation['num_zoom_in'].append(num_zoom_in)
                    annotation['num_zoom_out'].append(num_zoom_out)
                    annotation['search_mode'] = 2
                else:
                    annotation['search_mode'] = 3

    annotation['searched_bbox'] = [node.state.bbox for node in searched_nodes]

    # Evidence memory: compile searched_nodes into montage for enhanced answering
    evidence_image = None
    if evidence_compiler is not None and searched_nodes:
        print(f"[EvidenceMemory] compiler present, {len(searched_nodes)} nodes, calling _compile_evidence...")
        try:
            evidence_image = _compile_evidence(
                evidence_compiler, image_pil, question, searched_nodes, annotation, debug_recorder
            )
            print(f"[EvidenceMemory] result: {'montage loaded' if evidence_image else 'None (no montage)'}")
        except Exception as e:
            import traceback
            print(f"[EvidenceMemory] compile failed: {e}")
            traceback.print_exc()
            evidence_image = None
    elif evidence_compiler is None:
        print("[EvidenceMemory] evidence_compiler is None, skipping")
    else:
        print("[EvidenceMemory] no searched_nodes, skipping")

    # Use evidence montage if available, otherwise original image
    final_image = evidence_image if evidence_image is not None else image_pil
    final_nodes = [] if evidence_image is not None else searched_nodes

    answer_type = annotation.get('answer_type', 'free_form')
    # For vstar
    if answer_type == "logits_match":
        option_choose = zoom_model.multiple_choices_inference(final_image, question, options, final_nodes)
        if debug_recorder is not None:
            debug_recorder.record_final(annotation, image_pil, searched_nodes, option_choose)
        return option_choose
    elif answer_type == "free_form":
        if question_free_form:
            response = zoom_model.free_form_using_nodes(final_image, question_free_form, final_nodes)
        else:
            response = zoom_model.free_form_using_nodes(final_image, question, final_nodes)
        if debug_recorder is not None:
            debug_recorder.record_final(annotation, image_pil, searched_nodes, response)
        return response
    # For hr-bench
    elif answer_type == "option_list":
        answers = []
        for option_str in options:
            question_input = format_question(question, option_str)
            answers.append(zoom_model.free_form_using_nodes(final_image, question_input, final_nodes))
        if debug_recorder is not None:
            debug_recorder.record_final(annotation, image_pil, searched_nodes, answers)
        return answers
    # For mme-realworld
    elif answer_type == "Multiple Choice":
        question_input = format_question_multichoice(question, options)
        response = zoom_model.free_form_using_nodes(final_image, question_input, final_nodes)
        if debug_recorder is not None:
            debug_recorder.record_final(annotation, image_pil, searched_nodes, response)
        return response
    elif answer_type == "option_single":
        question_input = format_question_new(question, options)
        response = zoom_model.free_form_using_nodes(final_image, question_input, final_nodes)
        if debug_recorder is not None:
            debug_recorder.record_final(annotation, image_pil, searched_nodes, response)
        return response
    else:
        raise NotImplementedError


def _compile_evidence(evidence_compiler, image_pil, question, searched_nodes, annotation, debug_recorder):
    """Build evidence proposals from searched_nodes and compile via evidence_compiler."""
    from cvsearch.evidence_memory import EvidenceProposal, TargetSpec

    # Build targets from annotation
    targets_text = annotation.get('targets', []) or []
    targets = [TargetSpec(target_id=f"target_{i}", phrase=t) for i, t in enumerate(targets_text)]
    if not targets:
        targets = [TargetSpec(target_id="target_0", phrase="target")]

    # Convert searched_nodes to EvidenceProposals
    proposals = []
    for index, node in enumerate(searched_nodes):
        if getattr(node, "is_root", False):
            continue
        # Use node.target_phrase set by CVSearch, fallback to index-based mapping
        node_phrase = getattr(node, "target_phrase", None)
        if node_phrase:
            target = next((t for t in targets if t.phrase == node_phrase), None)
            if target is None:
                target = TargetSpec(target_id=f"target_{node_phrase}", phrase=node_phrase)
        else:
            target = targets[min(index, len(targets) - 1)]
        proposals.append(
            EvidenceProposal(
                target=target,
                source_name=getattr(node, "search_source", "cvsearch"),
                source_id=f"searched_{index:02d}_{getattr(node, 'id', index)}",
                box=tuple(float(v) for v in node.state.bbox),
                score=float(getattr(node, "answering_confidence", getattr(node, "posterior_score", 0.0)) or 0.0),
                metadata={
                    "node_id": getattr(node, "id", None),
                    "depth": getattr(node, "depth", None),
                    "search_source": getattr(node, "search_source", None),
                },
            )
        )

    if not proposals:
        return None

    # Build context with debug info
    context = {"question": question, "annotation": annotation}
    if debug_recorder is not None:
        sample_dir = getattr(debug_recorder, "sample_dir", None) or getattr(debug_recorder, "output_dir", None)
        if sample_dir:
            from pathlib import Path as _Path
            from cvsearch.debug.artifacts import ArtifactStore
            sample_dir = _Path(sample_dir)
            context["artifact_store"] = ArtifactStore(sample_dir)
            context["evidence_montage_path"] = str(sample_dir / "12_evidence_memory_montage.jpg")
            context["evidence_model_input_path"] = str(sample_dir / "12_evidence_model_input.jpg")

    # Compile
    artifact = evidence_compiler.compile(
        image_pil,
        question,
        proposals=proposals,
        targets=targets,
        context=context,
    )

    # Load montage image if available
    if artifact.montage and artifact.montage.model_input_path:
        montage_path = artifact.montage.model_input_path
        try:
            return Image.open(montage_path).convert("RGB")
        except Exception:
            pass

    return None


def get_cvsearch_leaf_batch_response(
        sam_model,
        zoom_model,
        nlp_model,
        annotation,
        ic_examples,
        decomposed_question_template,
        answering_confidence_threshold_upper,
        answering_confidence_threshold_lower,
        fast_threshold,
        pop_limit,
        threshold_descrease,
        image_folder: str = None,
        debug_recorder=None,
        evidence_compiler=None,
        leaf_depth_range: tuple = (1, 3),
):
    """Leaf-batch evidence pipeline: build the search tree, collect all leaf
    nodes as proposals, and run evidence_compiler without recursive search.

    The existing ``get_cvsearch_response`` is unchanged.  This entry point
    replaces the ``semantic_guide_search`` step entirely — the tree provides
    spatial decomposition and evidence_compiler handles selection.
    """
    from cvsearch.evidence_memory import EvidenceProposal, TargetSpec
    from cvsearch.evidence_memory.leaf_collector import collect_leaf_proposals

    keep_threshold_mllm_primary = 0.15
    tree_depth = 3

    input_image = annotation['input_image']
    if image_folder is not None:
        input_image = os.path.join(image_folder, input_image)
    image_pil = Image.open(input_image).convert('RGB')
    if debug_recorder is not None:
        debug_recorder.start_sample(annotation, image_pil, input_image)

    question = annotation['question']
    options = annotation.get('options', None)
    answer_type = annotation.get('answer_type', 'option_single')

    # Build feature-map and search tree (same as get_cvsearch_response).
    with torch.inference_mode():
        backbone_out, processed_results, target_id = sam_model.batch_inference(
            image_pil, ["target"]
        )
    image_features_batch = backbone_out['vision_features']
    if isinstance(image_features_batch, torch.Tensor):
        feat = image_features_batch.detach().cpu().float().numpy()
    else:
        feat = image_features_batch
    feat = feat.squeeze(0)
    del backbone_out, image_features_batch

    builder = ConstrainedTreeBuilder(
        feat, n_atoms=600, pos_weight=3.5, split_threshold=0.3,
        keep_threshold=keep_threshold_mllm_primary,
    )
    tree_dict = builder.build_tree(max_depth=tree_depth, min_splits=4, max_splits=8)
    feat_shape = feat.shape
    image_tree = AdaptiveImageTree(image_pil, tree_dict, feat_shape)
    if debug_recorder is not None:
        debug_recorder.record_tree("leaf_batch_tree", image_tree)

    # Build target list from annotation.
    targets_text = annotation.get('targets', []) or []
    if not targets_text:
        # Fall back to NLP extraction mirroring get_cvsearch_response.
        targets_text = extract_visual_objects(nlp_model, question) or []
    targets = [TargetSpec(target_id=f"target_{i}", phrase=t) for i, t in enumerate(targets_text)]
    if not targets:
        targets = [TargetSpec(target_id="target_0", phrase="target")]

    # Collect leaf proposals — no recursive search needed.
    proposals = collect_leaf_proposals(image_tree, targets, depth_range=leaf_depth_range)

    if not proposals or evidence_compiler is None:
        # Graceful fallback: answer from the full image with no evidence.
        img_w, img_h = image_pil.size
        state = NodeState(image_pil, [0, 0, img_w, img_h])
        root_node = NodeA(state)
        root_node.is_root = True
        question_input = format_question_new(question, options)
        response = zoom_model.free_form_using_nodes(image_pil, question_input, [root_node])
        if debug_recorder is not None:
            debug_recorder.record_final(annotation, image_pil, [], response)
        return response

    # Build evidence context.
    context = {"question": question, "annotation": annotation}
    if debug_recorder is not None:
        sample_dir = getattr(debug_recorder, "sample_dir", None) or getattr(debug_recorder, "output_dir", None)
        if sample_dir:
            from pathlib import Path as _Path
            from cvsearch.debug.artifacts import ArtifactStore
            sample_dir = _Path(sample_dir)
            context["artifact_store"] = ArtifactStore(sample_dir)
            context["evidence_montage_path"] = str(sample_dir / "12_evidence_memory_montage.jpg")
            context["evidence_model_input_path"] = str(sample_dir / "12_evidence_model_input.jpg")

    # Compile evidence (window building + keeping + layout).
    artifact = evidence_compiler.compile(
        image_pil,
        question,
        proposals=proposals,
        targets=targets,
        context=context,
    )

    # Use the montage image if available; otherwise fall back to full image.
    evidence_image = None
    if artifact.montage and artifact.montage.model_input_path:
        try:
            evidence_image = Image.open(artifact.montage.model_input_path).convert("RGB")
        except Exception:
            pass

    final_image = evidence_image if evidence_image is not None else image_pil
    img_w, img_h = image_pil.size
    state = NodeState(image_pil, [0, 0, img_w, img_h])
    root_node = NodeA(state)
    root_node.is_root = True
    final_nodes = [] if evidence_image is not None else [root_node]

    answer_type = annotation.get('answer_type', 'free_form')
    if answer_type == "logits_match":
        response = zoom_model.multiple_choices_inference(final_image, question, options, final_nodes)
    elif answer_type == "free_form":
        response = zoom_model.free_form_using_nodes(final_image, question, final_nodes)
    elif answer_type == "option_list":
        answers = []
        for option_str in options:
            question_input = format_question(question, option_str)
            answers.append(zoom_model.free_form_using_nodes(final_image, question_input, final_nodes))
        response = answers
    elif answer_type == "Multiple Choice":
        question_input = format_question_multichoice(question, options)
        response = zoom_model.free_form_using_nodes(final_image, question_input, final_nodes)
    elif answer_type == "option_single":
        question_input = format_question_new(question, options)
        response = zoom_model.free_form_using_nodes(final_image, question_input, final_nodes)
    else:
        raise NotImplementedError(f"Unsupported answer_type: {answer_type}")

    if debug_recorder is not None:
        debug_recorder.record_final(annotation, image_pil, [], response)
    return response


def process_sam_result(processed_results, target_id, is_second_search):
    """
    process SAM result
    """
    sam_success_flags = []
    sam_bboxes = []
    for t_id in target_id:
        # 1. boxes and scores
        boxes = processed_results[t_id]["boxes"].float().cpu().numpy()
        scores = processed_results[t_id]["scores"].float().cpu().numpy()
        # 2. Check if there are valid bbox
        if boxes.size > 0 and boxes.ndim > 1 and boxes.shape[1] >= 4:
            if not is_second_search:
                min_x1 = np.min(boxes[:, 0])
                min_y1 = np.min(boxes[:, 1])
                max_x2 = np.max(boxes[:, 2])
                max_y2 = np.max(boxes[:, 3])
                merged_bbox = [int(min_x1), int(min_y1), int(max_x2), int(max_y2)]
                sam_bboxes.append(merged_bbox)
                sam_success_flags.append(1)
            else:
                current_target_all_boxes = []
                for box in boxes:
                    bbox_int = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                    current_target_all_boxes.append(bbox_int)

                sam_bboxes.extend(current_target_all_boxes)
                sam_success_flags.append(1)
        else:
            sam_success_flags.append(0)
            sam_bboxes.append([])

    return sam_success_flags, sam_bboxes

def crop_image_by_node(
        image_pil: Image.Image,
        node: dict,
):
    """
    Args:
        image_pil (PIL.Image)
        node (dict): bbox format (x1, x1, w, h)
    Returns:
        PIL.Image: Image Patch
    """
    orig_w, orig_h = image_pil.size
    if isinstance(node, dict):
        bbox = node['bbox']
    elif hasattr(node, 'bbox'):
        bbox = node.bbox
    elif hasattr(node, 'state') and hasattr(node.state, 'bbox'):
        bbox = node.state.bbox
    else:
        raise ValueError("Provided node does not contain valid bbox information.")
    x1, y1, w, h = bbox
    # 4. Feature Map -> Original Image
    oy1 = int(y1)
    ox1 = int(x1)
    oy2 = int(y1+h)
    ox2 = int(x1+w)
    # 5. PIL crop format: left, top, right, bottom
    crop_box = (
        max(0, ox1),  # left
        max(0, oy1),  # top
        min(orig_w, ox2),  # right
        min(orig_h, oy2)  # bottom
    )
    if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
        patch = image_pil.crop(crop_box)
        return patch, crop_box
    else:
        return None, None

def format_question(question, option_str):
    return question + '\n' + option_str + 'Answer the option letter directly.'

def format_question_new(question, option_str):
    return question + " Options:\n" + option_str + "\nSelect the best answer to the above multiple-choice question based on the image. Respond with only the letter of the correct option.\nThe best answer is:"

def format_question_multichoice(question, options):
    ret = question
    for o in options:
        ret += '\n'
        ret += o
    # This prompt is copied from the original paper of MME-RealWorld
    ret += '\nSelect the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option.\nThe best answer is:'
    return ret

def semantic_guide_search_dynamic_depth(
        zoom_model,
        pop_limit: Union[int, Callable],
        num_intervel: int,
        threshold_descrease: List[float],
        depth_limit: int,
        question: str,
        visual_cue: str,
        answering_confidence_threshold_lower: float,
        answering_confidence_threshold_upper: float,
        image_pil=None,
        image_tree=None,
        w_current: float = 0.4,
        w_child: float = 0.4,
        w_prior: float = 0.2,
        prior_pruning_threshold: float = 0.4,
        parent_verification_threshold: float = 0.0,
        high_confidence_bypass: float = 0.8,
        enable_parent_verification: bool = True,
        debug_recorder=None,
        debug_label=None,
) -> Tuple[List, int, bool]:
    # -------------------------------------------------------------------------
    # 0. Initialization and dynamic depth detection
    # -------------------------------------------------------------------------
    # Determine the maximum depth allowed for this search
    actual_max_depth = min(image_tree.max_depth, depth_limit)
    pop_num_limit = pop_limit(actual_max_depth) if callable(pop_limit) else pop_limit

    nodes_by_depth = {}
    queue = [image_tree.root]
    while queue:
        node = queue.pop(0)
        if 1 <= node.depth <= actual_max_depth:
            if node.depth not in nodes_by_depth:
                nodes_by_depth[node.depth] = []
            nodes_by_depth[node.depth].append(node)
            node.aggregated_confidence = -1.0
            node.posterior_score = -1.0
            node.fast_confidence = None

        if node.depth < actual_max_depth:
            queue.extend(node.children)

    total_pop = 0
    debug_trace_id = None
    if debug_recorder is not None:
        debug_trace_id = debug_recorder.start_search(debug_label or visual_cue, visual_cue, question, image_pil)

    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------
    def calc_existence_and_update_parent(node):
        if node.fast_confidence is None:
            if node.prior_prob > prior_pruning_threshold:
                existence = zoom_model.get_confidence_value([node], image_pil, confidence_type='existence',input_ele=visual_cue)
                node.fast_confidence = existence
                node.is_evaluated = True
            else:
                node.fast_confidence = -1.0
                node.is_evaluated = False

        if node.parent and hasattr(node.parent, 'aggregated_confidence'):
            node.parent.aggregated_confidence = max(node.parent.aggregated_confidence, node.fast_confidence)

    def calc_score_and_sort(nodes, use_child_info=True):
        valid_nodes_for_sorting = []
        for node in nodes:
            calc_existence_and_update_parent(node)
            if not getattr(node, 'is_evaluated', False):
                node.posterior_score = -999.0
                continue

            norm_fast_conf = (node.fast_confidence + 1.0) / 2.0
            norm_agg = (node.aggregated_confidence + 1.0) / 2.0

            if use_child_info:
                score = (w_current * norm_fast_conf) + (w_child * norm_agg) + (w_prior * node.prior_prob)
            else:
                sum_local = w_current + w_prior
                n_wc = w_current / sum_local if sum_local > 0 else 0.5
                n_wp = w_prior / sum_local if sum_local > 0 else 0.5
                score = (n_wc * norm_fast_conf) + (n_wp * node.prior_prob)

            node.posterior_score = score
            valid_nodes_for_sorting.append(node)
        return sorted(valid_nodes_for_sorting, key=lambda x: x.posterior_score, reverse=True)

    def execute_stage_search(Q, stage_name, start_pop_count, check_parent=False):
        pop_trace = []
        current_threshold = answering_confidence_threshold_upper
        temp_threshold_descrease = deepcopy(threshold_descrease)
        next_checkpoint = pop_num_limit
        last_step = 0.05
        local_pop = 0

        def validate_node(node, confidence):
            if not check_parent or not node.parent:
                return True, "No Check"

            if node.parent.fast_confidence is None:
                p_exist = zoom_model.get_confidence_value([node.parent], image_pil, confidence_type='existence',input_ele=visual_cue)
                node.parent.fast_confidence = p_exist
            parent_conf = node.parent.fast_confidence
            if parent_conf >= parent_verification_threshold:
                return True, f"Parent Confirmed ({parent_conf:.2f})"
            if confidence >= high_confidence_bypass:
                return True, f"Bypass (Child {confidence:.2f} >> Parent {parent_conf:.2f})"
            return False, f"Rejected (Child {confidence:.2f} & Parent {parent_conf:.2f})"

        while len(Q) > 0:
            cur_node = Q.pop(0)
            local_pop += 1
            ans_conf = zoom_model.get_confidence_value([cur_node], image_pil, confidence_type='answering', input_ele=question)
            cur_node.answering_confidence = ans_conf
            pop_trace.append(cur_node)
            if debug_recorder is not None:
                debug_recorder.record_search_step(debug_trace_id, stage_name, cur_node, ans_conf, current_threshold)
            # print(f"[{stage_name}] ID:{cur_node.id} | Ans:{ans_conf:.4f}")

            if ans_conf >= current_threshold:
                is_valid, reason = validate_node(cur_node, ans_conf)
                if is_valid:
                    # print(f"  -> {reason} >>> Hit! Node {cur_node.id}")
                    return True, [cur_node], local_pop
                # else:
                #     print(f"  -> {reason} (Searching next...)")

            if local_pop >= next_checkpoint:
                # print(f"--- {stage_name} Checkpoint reached. Adjusting Threshold... ---")
                step = 0.0
                if len(temp_threshold_descrease) > 0:
                    step = temp_threshold_descrease.pop(0)
                    last_step = step
                else:
                    step = last_step
                if step > 0:
                    current_threshold -= step
                    current_threshold = max(current_threshold, answering_confidence_threshold_lower)
                    # print(f"New Threshold: {current_threshold:.4f}")

                    candidates = [n for n in pop_trace if n.answering_confidence >= current_threshold]
                    if candidates:
                        candidates.sort(key=lambda x: x.answering_confidence, reverse=True)
                        for cand in candidates:
                            is_valid, reason = validate_node(cand, cand.answering_confidence)
                            if is_valid:
                                # print(f">>> {stage_name} Hit via Decay! Node {cand.id} (Reason: {reason})")
                                return True, [cand], local_pop
                            # else:
                            #     print(f"  [Decay Check] Node {cand.id} skipped: {reason}")
                next_checkpoint += num_intervel
                if current_threshold <= answering_confidence_threshold_lower:
                    # print(f"Threshold hit lower bound. Stopping {stage_name}.")
                    break

        # print(f"--- {stage_name} Search Exhausted. Final Check... ---")
        if pop_trace:
            final_cands = [n for n in pop_trace if n.answering_confidence >= answering_confidence_threshold_lower]
            if final_cands:
                final_cands.sort(key=lambda x: x.answering_confidence, reverse=True)
                for cand in final_cands:
                    is_valid, reason = validate_node(cand, cand.answering_confidence)
                    if is_valid:
                        # print(f">>> {stage_name} Hit via Final Check! Node {cand.id} (Reason: {reason})")
                        return True, [cand], local_pop
                    # else:
                    #     print(f"  [Final Check] Node {cand.id} skipped: {reason}")
        return False, [], local_pop

    # -------------------------------------------------------------------------
    # Main process: dynamic hierarchical search
    # -------------------------------------------------------------------------
    search_depths = sorted([d for d in nodes_by_depth.keys() if d > 1], reverse=True)
    for idx, depth in enumerate(search_depths):
        is_bottom_layer = (idx == 0)
        stage_name = f"Depth {depth}"
        # print(f"\n=== Stage {idx + 1}: Searching {stage_name} (Total {len(nodes_by_depth[depth])} nodes) ===")

        current_use_child_info = not is_bottom_layer
        current_check_parent = is_bottom_layer and enable_parent_verification

        Q = calc_score_and_sort(nodes_by_depth[depth], use_child_info=current_use_child_info)

        success, res, count = execute_stage_search(
            Q, stage_name, total_pop, check_parent=current_check_parent
        )

        total_pop += count
        if success:
            if debug_recorder is not None:
                debug_recorder.finish_search(debug_trace_id, True, res)
            return res, total_pop, True

    # -------------------------------------------------------------------------
    # Stage Final: Depth 1
    # -------------------------------------------------------------------------
    if 1 in nodes_by_depth:
        # print(f"\n=== Final Stage: Searching Depth 1 (Total {len(nodes_by_depth[1])} nodes) ===")
        Q = calc_score_and_sort(nodes_by_depth[1], use_child_info=True)
        if Q:
            target = Q[0]
            total_pop += 1
            ans_conf = zoom_model.get_confidence_value([target], image_pil, confidence_type='answering',input_ele=question)
            target.answering_confidence = ans_conf
            if debug_recorder is not None:
                debug_recorder.record_search_step(debug_trace_id, "Depth 1", target, ans_conf, answering_confidence_threshold_lower)
            # print(f"[Depth 1] Best Node {target.id} | Ans: {ans_conf:.4f}")
            if ans_conf >= answering_confidence_threshold_lower:
                if debug_recorder is not None:
                    debug_recorder.finish_search(debug_trace_id, True, [target])
                return [target], total_pop, True

        all_d1 = sorted(nodes_by_depth[1], key=lambda x: getattr(x, 'posterior_score', -1), reverse=True)
        if debug_recorder is not None:
            debug_recorder.finish_search(debug_trace_id, False, all_d1[:1])
        return all_d1, total_pop, False

    if debug_recorder is not None:
        debug_recorder.finish_search(debug_trace_id, False, [])
    return [], total_pop, False


def get_direct_response(
        zoom_model,
        annotation,
        image_folder
):
    input_image = annotation['input_image']
    if image_folder is not None:
        input_image = os.path.join(image_folder, input_image)
    question = annotation['question']
    options = annotation.get('options', None)

    image_pil = Image.open(input_image).convert('RGB')

    # An empty list will conduct direct answering.
    searched_nodes = []
    answer_type = annotation.get('answer_type', 'free_form')
    # For vstar
    if answer_type == "logits_match":
        option_choose = zoom_model.multiple_choices_inference(image_pil, question, options, searched_nodes)
        return option_choose
    elif answer_type == "free_form":
        return zoom_model.free_form_using_nodes(image_pil, question, searched_nodes)
    # For hr-bench
    elif answer_type == "option_list":
        answers = []
        for option_str in options:
            question_input = format_question(question, option_str)
            answers.append(zoom_model.free_form_using_nodes(image_pil, question_input, searched_nodes))
        return answers
    elif answer_type == "option_single":
        question_input = format_question_new(question, options)
        response = zoom_model.free_form_using_nodes(image_pil, question_input, searched_nodes)
        return response
    elif answer_type == "Multiple Choice":
        question_input = format_question_multichoice(question, options)
        response = zoom_model.free_form_using_nodes(image_pil, question_input, searched_nodes)
        return response
    else:
        raise NotImplementedError
