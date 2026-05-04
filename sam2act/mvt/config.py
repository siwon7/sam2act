# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from yacs.config import CfgNode as CN

_C = CN()

_C.depth = 8
_C.img_size = 220
_C.add_proprio = True
_C.proprio_dim = 4
_C.add_lang = True
_C.lang_dim = 512
_C.lang_len = 77
_C.img_feat_dim = 3
_C.feat_dim = (72 * 3) + 2 + 2
_C.im_channels = 64
_C.attn_dim = 512
_C.attn_heads = 8
_C.attn_dim_head = 64
_C.activation = "lrelu"
_C.weight_tie_layers = False
_C.attn_dropout = 0.1
_C.decoder_dropout = 0.0
_C.img_patch_size = 11
_C.final_dim = 64
_C.self_cross_ver = 1
_C.add_corr = True
_C.norm_corr = False
_C.add_pixel_loc = True
_C.add_depth = True
_C.rend_three_views = False
_C.use_point_renderer = False
_C.pe_fix = True
_C.feat_ver = 0
_C.wpt_img_aug = 0.01
_C.inp_pre_pro = True
_C.inp_pre_con = True
_C.cvx_up = False
_C.xops = False
_C.rot_ver = 0
_C.num_rot = 72
_C.stage_two = False
_C.st_sca = 4
_C.st_wpt_loc_aug = 0.05
_C.st_wpt_loc_inp_no_noise = False
_C.img_aug_2 = 0.0

_C.ifSAM2 = True
_C.lora_finetune = True
_C.lora_r = 16
_C.ifsep = False
_C.resize_rgb = True
_C.use_memory = False
_C.num_maskmem = 7
_C.graph_node_classes = 0

_C.enable_task_memory_routing = False
_C.enable_dual_memory = False
_C.enable_event_memory = False
_C.enable_event_write = False
_C.enable_local_event_fusion = False
_C.enable_adaptive_event_prune = False
_C.enable_heatmap_event_features = False
_C.edge_bias_enabled = False
_C.edge_bias_lambda = 0.0
_C.edge_bias_temporal_scale = 0.0
_C.edge_bias_revisit_scale = 0.0
_C.edge_bias_transition_scale = 0.0
_C.edge_bias_revisit_sigma = 0.25
_C.edge_bias_ref_match_threshold = 0.05
_C.edge_bias_transition_hop = 1

# Stage1: Multi-peak heatmap labels
_C.use_multipeak = False
_C.multipeak_mode = "both"  # "intra", "cross", "both"
_C.multipeak_targets_json = ""  # path to precomputed JSON (if empty, compute on-the-fly)
_C.multipeak_cluster_radius = 0.04
_C.multipeak_max_peaks = 5

# Stage2: Graph-based peak selector
_C.use_graph_peak_select = False
_C.graph_node_embed_dim = 64
_C.graph_peak_topk = 3
_C.graph_peak_select_loss_weight = 0.5
_C.graph_transition_loss_weight = 0.1
_C.graph_contrastive_loss_weight = 0.05
_C.graph_contrastive_temperature = 0.1
_C.graph_contrastive_pos_radius = 0.03
_C.graph_peak_insert_gt_train = True
_C.graph_peak_positive_radius = 0.05
_C.graph_peak_nms_dist = 0.05

# Stage2 V10 candidate interface
# top1: original/V9 single crop behavior
# selector: V10a, graph selector chooses one Stage1 candidate crop
# kcrop: V10b, run Stage2 on K candidate crops and keep the selected branch
_C.stage2_candidate_mode = "top1"
_C.stage2_candidate_train_crop = "gt"  # "gt", "selector", "nearest_gt"
_C.stage2_kcrop_train_pick = "target"  # "target", "selector"
_C.stage2_candidate_insert_gt_train = True

# V10c memory write formatting for SAM2Act+ memory paths.
# stage1 keeps raw logits; topk_soft/selected write sparse normalized masks.
_C.stage2_memory_write_mode = "stage1"
_C.stage2_memory_write_topk = 3
_C.stage2_memory_write_temperature = 0.25
_C.stage2_memory_write_sigma = 1.5

_C.sam2_config = '/configs/sam2.1/sam2.1_hiera_b+'
_C.sam2_ckpt = './mvt/sam2_train/checkpoints/sam2.1_hiera_base_plus.pt'

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()
