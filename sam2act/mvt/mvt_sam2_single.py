# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from math import ceil

import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange, repeat

import sam2act.mvt.utils as mvt_utils
from sam2act.mvt.attn import (
    Conv2DBlock,
    Conv2DUpsampleBlock,
    PreNorm,
    Attention,
    cache_fn,
    DenseBlock,
    FeedForward,
    FixedPositionalEncoding,
    Fusion_up,
    act_layer
)
from sam2act.mvt.raft_utils import ConvexUpSample
from sam2act.utils.memorybench_role_graph import should_store_persistent_anchor



def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(
            m.weight, nonlinearity="relu"
        )
        nn.init.zeros_(m.bias)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MVT_SAM2_Single(nn.Module):
    def __init__(
        self,
        depth,
        img_size,
        add_proprio,
        proprio_dim,
        add_lang,
        lang_dim,
        lang_len,
        img_feat_dim,
        feat_dim,
        im_channels,
        attn_dim,
        attn_heads,
        attn_dim_head,
        activation,
        weight_tie_layers,
        attn_dropout,
        decoder_dropout,
        img_patch_size,
        final_dim,
        self_cross_ver,
        add_corr,
        norm_corr,
        add_pixel_loc,
        add_depth,
        rend_three_views,
        use_point_renderer,
        pe_fix,
        feat_ver,
        wpt_img_aug,
        inp_pre_pro,
        inp_pre_con,
        cvx_up,
        xops,
        rot_ver,
        num_rot,
        sam2,
        ifsep,
        resize_rgb,
        lora_finetune,
        lora_r,
        ifSAM2,
        rank,
        sam2_config,
        sam2_ckpt,
        use_memory,
        num_maskmem,
        graph_node_classes,
        phase_graph_num_classes,
        role_graph_enabled,
        role_graph_num_classes,
        role_graph_hidden_dim,
        role_graph_bias_scale,
        anchor_use_bias_scale,
        role_contrastive_dim,
        memory_gate_enabled=False,
        memory_gate_mode="none",
        memory_gate_use_text=True,
        memory_gate_use_task=False,
        memory_gate_hidden_dim=128,
        memory_gate_task_cond_dim=0,
        persistent_anchor_enabled=False,
        persistent_anchor_max_steps=2,
        persistent_anchor_prepend=True,
        renderer_device="cuda:0",
        renderer=None,
        no_feat=False,
    ):
        """MultiView Transfomer

        :param depth: depth of the attention network
        :param img_size: number of pixels per side for rendering
        :param renderer_device: device for placing the renderer
        :param add_proprio:
        :param proprio_dim:
        :param add_lang:
        :param lang_dim:
        :param lang_len:
        :param img_feat_dim:
        :param feat_dim:
        :param im_channels: intermediate channel size
        :param attn_dim:
        :param attn_heads:
        :param attn_dim_head:
        :param activation:
        :param weight_tie_layers:
        :param attn_dropout:
        :param decoder_dropout:
        :param img_patch_size: intial patch size
        :param final_dim: final dimensions of features
        :param self_cross_ver:
        :param add_corr:
        :param norm_corr: wether or not to normalize the correspondece values.
            this matters when pc is outide -1, 1 like for the two stage mvt
        :param add_pixel_loc:
        :param add_depth:
        :param rend_three_views: True/False. Render only three views,
            i.e. top, right and front. Overwrites other configurations.
        :param use_point_renderer: whether to use the point renderer or not
        :param pe_fix: matter only when add_lang is True
            Either:
                True: use position embedding only for image tokens
                False: use position embedding for lang and image token
        :param feat_ver: whether to max pool final features or use soft max
            values using the heamtmap
        :param wpt_img_aug: how much noise is added to the wpt_img while
            training, expressed as a percentage of the image size
        :param inp_pre_pro: whether or not we have the intial input
            preprocess layer. this was used in peract but not not having it has
            cost advantages. if false, we replace the 1D convolution in the
            orginal design with identity
        :param inp_pre_con: whether or not the output of the inital
            preprocessing layer is concatenated with the ouput of the
            upsampling layer for the "final" layer
        :param cvx_up: whether to use learned convex upsampling
        :param xops: whether to use xops or not
        :param rot_ver: version of the rotation prediction network
            Either:
                0: same as peract, independent discrete xyz predictions
                1: xyz prediction dependent on one another
        :param num_rot: number of discrete rotations per axis, used only when
            rot_ver is 1
        :param no_feat: whether to return features or not
        """

        super().__init__()
        self.depth = depth
        self.img_feat_dim = img_feat_dim
        self.img_size = img_size
        self.add_proprio = add_proprio
        self.proprio_dim = proprio_dim
        self.add_lang = add_lang
        self.lang_dim = lang_dim
        self.lang_len = lang_len
        self.im_channels = im_channels
        self.img_patch_size = img_patch_size
        self.final_dim = final_dim
        self.attn_dropout = attn_dropout
        self.decoder_dropout = decoder_dropout
        self.self_cross_ver = self_cross_ver
        self.add_corr = add_corr
        self.norm_corr = norm_corr
        self.add_pixel_loc = add_pixel_loc
        self.add_depth = add_depth
        self.pe_fix = pe_fix
        self.feat_ver = feat_ver
        self.wpt_img_aug = wpt_img_aug
        self.inp_pre_pro = inp_pre_pro
        self.inp_pre_con = inp_pre_con
        self.cvx_up = cvx_up
        self.use_point_renderer = use_point_renderer
        self.rot_ver = rot_ver
        self.num_rot = num_rot
        self.no_feat = no_feat
        self.rend_three_views = rend_three_views

        self.sam2 = sam2
        self.feat_img_dim = 48
        self.sam_img_dim = 48
        self.ifsep = ifsep
        self.resize_rgb = resize_rgb
        self.lora_finetune = lora_finetune
        self.ifSAM2 = ifSAM2
        self.rank = rank
        self.use_memory = use_memory
        self.num_maskmem = num_maskmem
        self.graph_node_classes = graph_node_classes
        self.phase_graph_num_classes = phase_graph_num_classes
        self.role_graph_enabled = role_graph_enabled
        self.role_graph_num_classes = role_graph_num_classes
        self.role_graph_hidden_dim = role_graph_hidden_dim
        self.role_graph_bias_scale = role_graph_bias_scale
        self.anchor_use_bias_scale = anchor_use_bias_scale
        self.role_contrastive_dim = role_contrastive_dim
        self.memory_gate_enabled = memory_gate_enabled
        self.memory_gate_mode = memory_gate_mode
        self.memory_gate_use_text = memory_gate_use_text
        self.memory_gate_use_task = memory_gate_use_task
        self.memory_gate_hidden_dim = memory_gate_hidden_dim
        self.memory_gate_task_cond_dim = memory_gate_task_cond_dim
        self.persistent_anchor_enabled = persistent_anchor_enabled
        self.persistent_anchor_max_steps = persistent_anchor_max_steps
        self.persistent_anchor_prepend = persistent_anchor_prepend

        self.curr_obs_idx = 0
        self.memory_gate_context = None
        self.latest_eval_role_tag = 0
        self.role_graph_logits_runtime = []
        self.visit_mode_logits_runtime = []
        self.phase_graph_logits_runtime = []
        self.role_ref_logits_runtime = []
        self.anchor_use_logits_runtime = []
        self.role_contrast_embeddings_runtime = []
        self.role_contrast_labels_runtime = []

        self.memory_bank_multiview = [{} for _ in range(3)] if self.rend_three_views else [{} for _ in range(5)]
        self.anchor_memory_bank_multiview = [{} for _ in range(3)] if self.rend_three_views else [{} for _ in range(5)]
        self.memory_role_tags_multiview = [{} for _ in range(3)] if self.rend_three_views else [{} for _ in range(5)]
        self.anchor_role_tags_multiview = [{} for _ in range(3)] if self.rend_three_views else [{} for _ in range(5)]


        if self.cvx_up:
            assert not self.inp_pre_con, (
                "When using the convex upsampling, we do not concatenate"
                " features from input_preprocess to the features used for"
                " prediction"
            )

        # print(f"MVT Vars: {vars(self)}")

        assert not renderer is None
        self.renderer = renderer
        self.num_img = self.renderer.num_img

        # patchified input dimensions
        spatial_size = img_size // self.img_patch_size  # 128 / 8 = 16

        if self.add_proprio:
            # 64 img features + 64 proprio features
            self.input_dim_before_seq = self.im_channels * 2
        else:
            self.input_dim_before_seq = self.im_channels

        # learnable positional encoding
        if add_lang:
            lang_emb_dim, lang_max_seq_len = lang_dim, lang_len
        else:
            lang_emb_dim, lang_max_seq_len = 0, 0
        self.lang_emb_dim = lang_emb_dim
        self.lang_max_seq_len = lang_max_seq_len

        if self.pe_fix:
            num_pe_token = spatial_size**2 * self.num_img
        else:
            num_pe_token = lang_max_seq_len + (spatial_size**2 * self.num_img)
        self.pos_encoding = nn.Parameter(
            torch.randn(
                1,
                num_pe_token,
                self.input_dim_before_seq,
            )
        )

        inp_img_feat_dim = self.img_feat_dim
        if self.add_corr:
            inp_img_feat_dim += 3
        if self.add_pixel_loc:
            inp_img_feat_dim += 3
            self.pixel_loc = torch.zeros(
                (self.num_img, 3, self.img_size, self.img_size)
            )
            self.pixel_loc[:, 0, :, :] = (
                torch.linspace(-1, 1, self.num_img).unsqueeze(-1).unsqueeze(-1)
            )
            self.pixel_loc[:, 1, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(-1)
            )
            self.pixel_loc[:, 2, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(0)
            )
        if self.add_depth:
            inp_img_feat_dim += 1

        # img input preprocessing encoder
        if self.inp_pre_pro:
            self.input_preprocess = Conv2DBlock(
                inp_img_feat_dim,
                self.im_channels,
                kernel_sizes=1,
                strides=1,
                norm=None,
                activation=activation,
            )
            inp_pre_out_dim = self.im_channels
        else:
            # identity
            self.input_preprocess = lambda x: x
            inp_pre_out_dim = inp_img_feat_dim

        if self.add_proprio:
            # proprio preprocessing encoder
            self.proprio_preprocess = DenseBlock(
                self.proprio_dim,
                32,
                norm="group",
                activation=activation,
            )

        self.patchify = Conv2DBlock(
            7 if self.ifsep else 10,
            self.feat_img_dim,
            kernel_sizes=self.img_patch_size,
            strides=self.img_patch_size,
            norm="group",
            activation=activation,
            padding=0,
        )

        # lang preprocess
        if self.add_lang:
            self.lang_preprocess = DenseBlock(
                lang_emb_dim,
                self.im_channels * 2,
                norm="group",
                activation=activation,
            )

        
        if self.sam_img_dim != 0:
            self.fusion_up = Fusion_up(
                256,
                self.sam_img_dim,
                kernel_sizes=1,
                strides=1,
                norm=None,
                activation=activation,
            )
            
            self.fusion = Conv2DBlock(
                256,
                self.sam_img_dim,
                kernel_sizes=1,
                strides=1,
                norm=None,
                activation=activation,
            )


        self.fc_bef_attn = DenseBlock(
            self.input_dim_before_seq,
            attn_dim,
            norm=None,
            activation=None,
        )

        # self.input_dim_before_seq = 256

        self.fc_aft_attn = DenseBlock(
            attn_dim,
            self.input_dim_before_seq,
            norm=None,
            activation=None,
        )

        self.memory_gate_dim = getattr(self.sam2, "mem_dim", 128)
        if self.memory_gate_enabled:
            if self.memory_gate_use_text and self.add_lang:
                self.memory_gate_text_proj = nn.Sequential(
                    nn.Linear(self.lang_emb_dim, self.memory_gate_hidden_dim),
                    act_layer(activation),
                    nn.Linear(self.memory_gate_hidden_dim, self.memory_gate_dim),
                )
            else:
                self.memory_gate_text_proj = None

            if self.memory_gate_use_task and self.memory_gate_task_cond_dim > 0:
                self.memory_gate_task_proj = nn.Sequential(
                    nn.Linear(self.memory_gate_task_cond_dim, self.memory_gate_hidden_dim),
                    act_layer(activation),
                    nn.Linear(self.memory_gate_hidden_dim, self.memory_gate_dim),
                )
            else:
                self.memory_gate_task_proj = None
        else:
            self.memory_gate_text_proj = None
            self.memory_gate_task_proj = None

        if self.persistent_anchor_enabled:
            self.persistent_anchor_tpos_enc = nn.Parameter(
                torch.zeros(1, 1, self.memory_gate_dim)
            )
        else:
            self.register_parameter("persistent_anchor_tpos_enc", None)

        self.role_query_input_dim = self.input_dim_before_seq
        if self.role_graph_enabled:
            self.role_query_proj = nn.Sequential(
                nn.Linear(self.role_query_input_dim, self.role_graph_hidden_dim),
                act_layer(activation),
                nn.Linear(self.role_graph_hidden_dim, self.role_graph_hidden_dim),
            )
            self.role_graph_head = nn.Linear(
                self.role_graph_hidden_dim, self.role_graph_num_classes
            )
            self.visit_mode_head = nn.Linear(self.role_graph_hidden_dim, 1)
            if self.phase_graph_num_classes > 0:
                self.phase_head = nn.Linear(
                    self.role_graph_hidden_dim, self.phase_graph_num_classes
                )
            else:
                self.phase_head = None
            self.role_ref_head = nn.Linear(
                self.role_graph_hidden_dim, self.role_graph_num_classes
            )
            self.anchor_use_head = nn.Linear(self.role_graph_hidden_dim, 1)
            self.role_contrast_proj = nn.Linear(
                self.memory_gate_dim, self.role_contrastive_dim
            )
            self.role_query_proj.apply(initialize_weights)
            self.role_graph_head.apply(initialize_weights)
            self.visit_mode_head.apply(initialize_weights)
            if self.phase_head is not None:
                self.phase_head.apply(initialize_weights)
            self.role_ref_head.apply(initialize_weights)
            self.anchor_use_head.apply(initialize_weights)
            self.role_contrast_proj.apply(initialize_weights)
        else:
            self.role_query_proj = None
            self.role_graph_head = None
            self.visit_mode_head = None
            self.phase_head = None
            self.role_ref_head = None
            self.anchor_use_head = None
            self.role_contrast_proj = None

        get_attn_attn = lambda: PreNorm(
            attn_dim,
            Attention(
                attn_dim,
                heads=attn_heads,
                dim_head=attn_dim_head,
                dropout=attn_dropout,
                use_fast=xops,
            ),
        )
        get_attn_ff = lambda: PreNorm(attn_dim, FeedForward(attn_dim))
        get_attn_attn, get_attn_ff = map(cache_fn, (get_attn_attn, get_attn_ff))
        # self-attention layers
        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}
        attn_depth = depth

        for _ in range(attn_depth):
            self.layers.append(
                nn.ModuleList([get_attn_attn(**cache_args), get_attn_ff(**cache_args)])
            )

        # self.channel_up = Conv2DBlock(
        #     self.input_dim_before_seq,
        #     256,
        #     kernel_sizes=1,
        #     strides=1,
        #     norm=None,
        #     activation=activation,
        # )

        if cvx_up:
            # self.up0 = ConvexUpSample(
            #     in_dim=self.input_dim_before_seq,
            #     out_dim=1,
            #     up_ratio=self.img_patch_size,
            # )

            # multi-resolution upsampling

            self.up0 = torch.nn.Sequential(
                ConvexUpSample(
                    in_dim=self.input_dim_before_seq,
                    out_dim=64,
                    up_ratio=2,
                ),
                LayerNorm2d(64),
                act_layer(activation),
                ConvexUpSample(
                    in_dim=64,
                    out_dim=32,
                    up_ratio=2,
                ),
                LayerNorm2d(32),
                act_layer(activation),
                torch.nn.Upsample(scale_factor=(self.img_size // 2) / (self.sam2.image_size // 4), mode="bilinear", align_corners=False),
                ConvexUpSample(
                    in_dim=32,
                    out_dim=1,
                    up_ratio=2,
                ),
            )
        else:
            self.up0 = Conv2DUpsampleBlock(
                self.input_dim_before_seq,
                self.im_channels,
                kernel_sizes=self.img_patch_size,
                strides=self.img_patch_size,
                norm=None,
                activation=activation,
                out_size=self.img_size,
            )

            if self.inp_pre_con:
                final_inp_dim = self.im_channels + inp_pre_out_dim
            else:
                final_inp_dim = self.im_channels

            # final layers
            self.final = Conv2DBlock(
                final_inp_dim,
                self.act_horizon*self.im_channels,
                kernel_sizes=3,
                strides=1,
                norm=None,
                activation=activation,
            )

            self.trans_decoder = Conv2DBlock(
                self.final_dim,
                1,
                kernel_sizes=3,
                strides=1,
                norm=None,
                activation=None,
            )

        if not self.no_feat:
            feat_fc_dim = 0
            feat_fc_dim += self.input_dim_before_seq
            if self.cvx_up:
                feat_fc_dim += self.input_dim_before_seq
            else:
                feat_fc_dim += self.final_dim

            def get_feat_fc(
                _feat_in_size,
                _feat_out_size,
                _feat_fc_dim=feat_fc_dim,
            ):
                """
                _feat_in_size: input feature size
                _feat_out_size: output feature size
                _feat_fc_dim: hidden feature size
                """
                layers = [
                    nn.Linear(_feat_in_size, _feat_fc_dim),
                    nn.ReLU(),
                    nn.Linear(_feat_fc_dim, _feat_fc_dim // 2),
                    nn.ReLU(),
                    nn.Linear(_feat_fc_dim // 2, _feat_out_size),
                ]
                feat_fc = nn.Sequential(*layers)
                return feat_fc

            feat_out_size = feat_dim

            if self.rot_ver == 0:
                self.feat_fc = get_feat_fc(
                    self.num_img * feat_fc_dim,
                    feat_out_size,
                )
                self.feat_fc.apply(initialize_weights)

            elif self.rot_ver == 1:
                assert self.num_rot * 3 <= feat_out_size
                feat_out_size_ex_rot = feat_out_size - (self.num_rot * 3)
                if feat_out_size_ex_rot > 0:
                    self.feat_fc_ex_rot = get_feat_fc(
                        self.num_img * feat_fc_dim, feat_out_size_ex_rot
                    )

                self.feat_fc_init_bn = nn.BatchNorm1d(self.num_img * feat_fc_dim)
                self.feat_fc_pe = FixedPositionalEncoding(
                    self.num_img * feat_fc_dim, feat_scale_factor=1
                )
                self.feat_fc_x = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)
                self.feat_fc_y = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)
                self.feat_fc_z = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)

            else:
                assert False

            if self.graph_node_classes > 0:
                self.graph_node_head = get_feat_fc(
                    self.num_img * feat_fc_dim,
                    self.graph_node_classes,
                )
                self.graph_node_head.apply(initialize_weights)

        if self.use_point_renderer:
            from point_renderer.rvt_ops import select_feat_from_hm
        else:
            from mvt.renderer import select_feat_from_hm
        global select_feat_from_hm

    def get_pt_loc_on_img(self, pt, dyn_cam_info):
        """
        transform location of points in the local frame to location on the
        image
        :param pt: (bs, np, 3)
        :return: pt_img of size (bs, np, num_img, 2)
        """
        pt_img = self.renderer.get_pt_loc_on_img(
            pt, fix_cam=True, dyn_cam_info=dyn_cam_info
        )
        return pt_img
    
    def sam2_image_encoder_forward(self, net, imgs):
        backbone_out = net.forward_image(imgs)
        # net._store_images_features(inference_state, imgs, backbone_out, len(imgs), num_views, start_frame_idx)
        _, vision_feats_all, vision_pos_embeds_all, _ = net._prepare_backbone_features(backbone_out)
        return vision_feats_all, vision_pos_embeds_all

    def _reset_role_graph_runtime(self):
        self.latest_eval_role_tag = 0
        self.role_graph_logits_runtime = []
        self.visit_mode_logits_runtime = []
        self.phase_graph_logits_runtime = []
        self.role_ref_logits_runtime = []
        self.anchor_use_logits_runtime = []
        self.role_contrast_embeddings_runtime = []
        self.role_contrast_labels_runtime = []

    def _build_role_query_outputs(self, vision_feats_per_view):
        if not self.role_graph_enabled:
            return None

        pooled_views = []
        for vision_feat in vision_feats_per_view:
            pooled_views.append(vision_feat.mean(dim=0).squeeze(0))
        query_input = torch.stack(pooled_views, dim=0).mean(dim=0, keepdim=True)
        role_hidden = self.role_query_proj(query_input)
        role_logits = self.role_graph_head(role_hidden)
        visit_mode_logits = self.visit_mode_head(role_hidden).squeeze(-1)
        phase_logits = (
            self.phase_head(role_hidden) if self.phase_head is not None else None
        )
        role_ref_logits = self.role_ref_head(role_hidden)
        anchor_use_logits = self.anchor_use_head(role_hidden).squeeze(-1)
        return {
            "hidden": role_hidden,
            "role_logits": role_logits,
            "visit_mode_logits": visit_mode_logits,
            "phase_logits": phase_logits,
            "role_ref_logits": role_ref_logits,
            "anchor_use_logits": anchor_use_logits,
        }

    def _build_memory_attn_bias(
        self,
        memory_entries,
        visit_mode_logits,
        phase_logits,
        role_ref_logits,
        anchor_use_logits,
        device,
        num_heads,
    ):
        if not self.role_graph_enabled or len(memory_entries) == 0:
            return None

        revisit_prob = torch.sigmoid(visit_mode_logits).view(-1)[0]
        late_phase_prob = revisit_prob.new_tensor(1.0)
        if phase_logits is not None and phase_logits.shape[-1] > 0:
            late_phase_prob = torch.softmax(phase_logits.view(-1), dim=0)[-1]
        role_ref_probs = torch.sigmoid(role_ref_logits).view(-1)
        anchor_use_prob = torch.sigmoid(anchor_use_logits).view(-1)[0]
        role_scale = self.role_graph_bias_scale
        anchor_scale = self.anchor_use_bias_scale

        token_biases = []
        for entry in memory_entries:
            entry_len = entry["memory"][0].shape[0]
            role_tag = int(entry["role_tag"])
            entry_bias = torch.zeros((), device=device)
            if 0 <= role_tag < role_ref_probs.shape[0]:
                entry_bias = entry_bias + late_phase_prob * role_scale * (2.0 * role_ref_probs[role_tag] - 1.0)
            if entry["is_anchor"]:
                entry_bias = entry_bias + late_phase_prob * anchor_scale * (2.0 * anchor_use_prob - 1.0)
            token_biases.append(entry_bias.repeat(entry_len))

        if not token_biases:
            return None

        key_bias = torch.cat(token_biases, dim=0)
        return key_bias.view(1, 1, 1, -1).expand(1, num_heads, 1, -1)
    
    def reset_memory_bank(self):
        self.memory_bank_multiview = [{} for _ in range(3)] if self.rend_three_views else [{} for _ in range(5)]
        self.anchor_memory_bank_multiview = [{} for _ in range(3)] if self.rend_three_views else [{} for _ in range(5)]
        self.memory_role_tags_multiview = [{} for _ in range(3)] if self.rend_three_views else [{} for _ in range(5)]
        self.anchor_role_tags_multiview = [{} for _ in range(3)] if self.rend_three_views else [{} for _ in range(5)]
        self.curr_obs_idx = 0

    def _get_anchor_memory_entries(self, view_idx, recent_keys):
        if not self.persistent_anchor_enabled:
            return []
        anchor_bank = self.anchor_memory_bank_multiview[view_idx]
        anchor_role_tags = self.anchor_role_tags_multiview[view_idx]
        anchor_entries = []
        for anchor_idx in sorted(anchor_bank.keys()):
            anchor_entries.append(
                {
                    "idx": anchor_idx,
                    "memory": anchor_bank[anchor_idx],
                    "role_tag": anchor_role_tags.get(anchor_idx, 0),
                    "is_anchor": True,
                }
            )
        return anchor_entries

    def _reset_memory_gate_context(self):
        self.memory_gate_context = None

    def _build_memory_gate_context(self, lang_emb=None, task_cond=None):
        self._reset_memory_gate_context()
        if not self.memory_gate_enabled:
            return

        gate_parts = []

        if self.memory_gate_text_proj is not None and lang_emb is not None:
            if lang_emb.dim() != 3:
                raise ValueError(
                    f"Expected lang_emb to have shape [bs, seq, dim], got {tuple(lang_emb.shape)}"
                )
            gate_parts.append(self.memory_gate_text_proj(lang_emb.mean(dim=1)))

        if self.memory_gate_task_proj is not None and task_cond is not None:
            if task_cond.dim() == 1:
                task_cond = task_cond.unsqueeze(-1)
            if task_cond.dim() != 2:
                raise ValueError(
                    f"Expected task_cond to have shape [bs, dim], got {tuple(task_cond.shape)}"
                )
            gate_parts.append(self.memory_gate_task_proj(task_cond))

        if gate_parts:
            self.memory_gate_context = torch.sigmoid(
                torch.stack(gate_parts, dim=0).sum(dim=0)
            )

    def _select_memory_gate(self, idx, device):
        if self.memory_gate_context is None:
            return None
        if idx < 0 or idx >= self.memory_gate_context.shape[0]:
            return None
        return self.memory_gate_context[idx: idx + 1].unsqueeze(0).to(device=device)

    def sam2_forward_with_memory(self, net, idx, num_views, feat_sizes):
        GPUdevice = self.rank
        image_embed_list = []
        vision_feats_per_view = []
        vision_pos_embeds_per_view = []

        for view_idx in range(num_views):
            vision_feats_per_view.append(
                self.vision_feats_all[-1][:, idx * num_views + view_idx, :]
                .unsqueeze(1)
                .clone()
                .cuda()
            )
            vision_pos_embeds_per_view.append(
                self.vision_pos_embeds_all[-1][:, idx * num_views + view_idx, :]
                .unsqueeze(1)
                .clone()
                .cuda()
            )

        role_query_outputs = self._build_role_query_outputs(vision_feats_per_view)
        if role_query_outputs is not None:
            self.role_graph_logits_runtime.append(role_query_outputs["role_logits"])
            if role_query_outputs["phase_logits"] is not None:
                self.phase_graph_logits_runtime.append(role_query_outputs["phase_logits"])
            self.visit_mode_logits_runtime.append(role_query_outputs["visit_mode_logits"])
            self.role_ref_logits_runtime.append(role_query_outputs["role_ref_logits"])
            self.anchor_use_logits_runtime.append(role_query_outputs["anchor_use_logits"])
            if not self.training:
                self.latest_eval_role_tag = int(
                    role_query_outputs["role_logits"].argmax(dim=-1).item()
                )

        num_heads = net.memory_attention.layers[0].cross_attn_image.num_heads

        for view_idx in range(num_views):
            vision_feats = [vision_feats_per_view[view_idx]]
            vision_pos_embeds = [vision_pos_embeds_per_view[view_idx]]

            memory_bank_list = self.memory_bank_multiview[view_idx]
            memory_role_tags = self.memory_role_tags_multiview[view_idx]
            memory_entries = []

            if len(memory_bank_list) > 0:
                num_mem = min(len(memory_bank_list), net.num_maskmem)
                recent_keys = set()

                for t_pos in range(1, num_mem + 1):
                    prev_idx = self.curr_obs_idx - t_pos
                    prev = memory_bank_list.get(prev_idx)
                    if prev is None:
                        continue
                    recent_keys.add(prev_idx)
                    memory_entries.append(
                        {
                            "idx": prev_idx,
                            "memory": prev,
                            "role_tag": memory_role_tags.get(prev_idx, 0),
                            "is_anchor": False,
                            "t_pos": t_pos,
                        }
                    )

                anchor_entries = self._get_anchor_memory_entries(view_idx, recent_keys)
                if anchor_entries:
                    if self.persistent_anchor_prepend:
                        memory_entries = anchor_entries + memory_entries
                    else:
                        memory_entries.extend(anchor_entries)

            if memory_entries:
                to_cat_memory = []
                to_cat_memory_pos_embed = []
                for entry in memory_entries:
                    feats = entry["memory"][0].cuda()
                    maskmem_enc = entry["memory"][1].cuda()
                    if entry["is_anchor"]:
                        if self.persistent_anchor_tpos_enc is not None:
                            maskmem_enc = maskmem_enc + self.persistent_anchor_tpos_enc
                    else:
                        maskmem_enc = maskmem_enc + net.maskmem_tpos_enc[entry["t_pos"] - 1]
                    to_cat_memory.append(feats)
                    to_cat_memory_pos_embed.append(maskmem_enc)

                memory_fused = torch.cat(to_cat_memory, dim=0)
                memory_pos_fused = torch.cat(to_cat_memory_pos_embed, dim=0)
                memory_gate = self._select_memory_gate(idx, memory_fused.device)
                if memory_gate is not None and self.memory_gate_mode in ("memory", "both"):
                    memory_fused = memory_fused * memory_gate

                attn_bias = None
                if role_query_outputs is not None:
                    attn_bias = self._build_memory_attn_bias(
                        memory_entries=memory_entries,
                        visit_mode_logits=role_query_outputs["visit_mode_logits"],
                        phase_logits=role_query_outputs["phase_logits"],
                        role_ref_logits=role_query_outputs["role_ref_logits"],
                        anchor_use_logits=role_query_outputs["anchor_use_logits"],
                        device=memory_fused.device,
                        num_heads=num_heads,
                    )

                pix_feat_with_mem = net.memory_attention(
                    curr=[vision_feats[-1]],
                    curr_pos=[vision_pos_embeds[-1]],
                    memory=memory_fused,
                    memory_pos=memory_pos_fused,
                    num_obj_ptr_tokens=0,
                    attn_bias=attn_bias,
                )
                if memory_gate is not None and self.memory_gate_mode in ("fusion", "both"):
                    pix_feat_with_mem = pix_feat_with_mem * memory_gate
            else:
                pix_feat_with_mem = vision_feats[-1]

            image_embed_with_mem = (
                pix_feat_with_mem.to(torch.float32)
                .permute(1, 2, 0)
                .view(1, -1, *feat_sizes[-1])
            )
            image_embed_list.append(image_embed_with_mem)
                            
            
        # merge all image features here
        image_embeds = torch.cat(image_embed_list, dim=0).to(GPUdevice)

        final_image_embeds = image_embeds.to(GPUdevice)

        return final_image_embeds

    def sam2_add_new_memory(self, net, hm, idx, num_views, feat_sizes, role_label=None):
        GPUdevice = self.rank
        if role_label is None:
            role_label_value = int(self.latest_eval_role_tag)
        elif isinstance(role_label, torch.Tensor):
            role_label_value = int(role_label.detach().view(-1)[0].item())
        else:
            role_label_value = int(role_label)

        contrast_view_embeds = []

        for view_idx in range(num_views):
                
            '''image encoder''' 

            vision_feats = [self.vision_feats_all[-1][:,idx*num_views+view_idx,:].unsqueeze(1).clone().cuda()]
            
            vision_pos_embeds= [self.vision_pos_embeds_all[-1][:,idx*num_views+view_idx,:].unsqueeze(1).clone().cuda()]
            
            hm_frame = hm[0, view_idx, :, :, :, :]
            
            '''encode memory'''
            high_res_multimasks = torch.nn.functional.interpolate(
                hm_frame.cuda(),
                size=(net.image_size, net.image_size),
                mode="bilinear",
                align_corners=False,
            )

            # new caluculated memory features
            maskmem_features, maskmem_pos_enc = net._encode_new_memory(
                current_vision_feats=vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_multimasks,
                object_score_logits=None,
                is_mask_from_pts=False,
                # if_sigmoid=True,
                )
            # dimension hint for your future use
            # maskmem_features: torch.Size([batch, 64, 64, 64])
            # maskmem_pos_enc: [torch.Size([batch, 64, 64, 64])].flatten(2).permute(2, 0, 1)
            
            maskmem_features = maskmem_features.to(device=GPUdevice, non_blocking=True)
            maskmem_pos_enc = maskmem_pos_enc[0].to(device=GPUdevice, non_blocking=True)

            memory = (
                maskmem_features.flatten(2)
                .permute(2, 0, 1)
                .reshape(-1, 1, self.memory_gate_dim)
            )
            memory_pos = (
                maskmem_pos_enc.flatten(2)
                .permute(2, 0, 1)
                .reshape(-1, 1, self.memory_gate_dim)
            )

            memory_bank_list = self.memory_bank_multiview[view_idx]

            memory_bank_list[self.curr_obs_idx] = [memory, memory_pos]
            self.memory_role_tags_multiview[view_idx][self.curr_obs_idx] = role_label_value
            if self.persistent_anchor_enabled:
                anchor_bank = self.anchor_memory_bank_multiview[view_idx]
                anchor_tags = self.anchor_role_tags_multiview[view_idx]
                if (
                    should_store_persistent_anchor(
                        role_label_value, self.role_graph_num_classes
                    )
                    and role_label_value not in anchor_bank
                ):
                    anchor_bank[role_label_value] = [memory, memory_pos]
                    anchor_tags[role_label_value] = role_label_value

            if self.role_graph_enabled and self.role_contrast_proj is not None:
                contrast_view_embeds.append(
                    self.role_contrast_proj(maskmem_features.mean(dim=(2, 3)))
                )

        if contrast_view_embeds:
            contrast_embed = torch.stack(contrast_view_embeds, dim=0).mean(dim=0)
            self.role_contrast_embeddings_runtime.append(contrast_embed)
            self.role_contrast_labels_runtime.append(role_label_value)

    def forward(
        self,
        img,
        proprio=None,
        lang_emb=None,
        wpt_local=None,
        rot_x_y=None,
        **kwargs,
    ):
        """
        :param img: tensor of shape (bs, num_img, img_feat_dim, h, w)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        :param img_aug: (float) magnitude of augmentation in rgb image
        :param rot_x_y: (bs, 2)
        """

        bs, num_img, img_feat_dim, h, w = img.shape
        num_pat_img = h // self.img_patch_size
        assert num_img == self.num_img
        # assert img_feat_dim == self.img_feat_dim
        assert h == w == self.img_size
        self._reset_role_graph_runtime()

        self._build_memory_gate_context(
            lang_emb=lang_emb,
            task_cond=kwargs.get("task_cond"),
        )

        img_raw = img.clone()
        img = img.view(bs * num_img, img_feat_dim, h, w)
        # preprocess
        # (bs * num_img, im_channels, h, w)
        d0 = self.input_preprocess(img)

        # if True:  # use SAM
        rgb_img = img[:,3:6]
        
        # rgb_img = img[:,3:6]
        # rgb_clip = True
        # if rgb_clip:
        #     rgb_img[rgb_img<0] = 0
        #     rgb_img[rgb_img>1] = 1 
            
        if self.resize_rgb:
            # import pdb;pdb.set_trace()
            rgb_img = F.interpolate(rgb_img, size=(256, 256), mode='bilinear', align_corners=True)
        
        feat_sizes = [(self.sam2.image_size // 4, self.sam2.image_size // 4), 
                        (self.sam2.image_size // 8, self.sam2.image_size // 8), 
                        (self.sam2.image_size // 16, self.sam2.image_size // 16)]
        
        self.sam2_vision_feats_all, self.sam2_vision_pos_embeds_all = self.sam2_image_encoder_forward(self.sam2, rgb_img)
        sam_out = self.sam2_vision_feats_all[-1].permute(1, 2, 0).view(bs*num_img, -1, *feat_sizes[-1])

        if num_pat_img == sam_out.shape[-1]:
            rgb_img = self.fusion(sam_out)
        else:
            rgb_img = self.fusion_up(sam_out)    ## c 256-> sam_img_dim
        rgb_img = (
            rgb_img.view(
            bs,
            num_img,
            self.sam_img_dim,
            num_pat_img,
            num_pat_img,
        ).transpose(1, 2).clone())   # torch.Size([bs, 48, 5, 20, 20])

        if self.ifsep:
            indices = [0, 1, 2, 6, 7, 8, 9]
            feat_img = img[:,indices]   
        else:
            feat_img = img

        feat_img = self.patchify(feat_img)   # conv2d  c 7 or 10 -> self.feat_img_dim
        feat_img = (
            feat_img.view(
                bs,
                num_img,
                self.feat_img_dim,
                num_pat_img,
                num_pat_img,
            ).transpose(1, 2).clone())   # torch.Size([bs, feat_img_dim :96 or 48, 5, 20, 20])
        _, _, _d, _h, _w = feat_img.shape
        if self.add_proprio:
            p = self.proprio_preprocess(proprio)  # [B,4] -> [B,32]    
            p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, _d, _h, _w)
            ins = torch.cat([rgb_img, feat_img, p], dim=1) if self.ifSAM2  \
                else torch.cat([feat_img, p], dim=1) # [B, 128, num_img, np, np]   96+32 or 48+48+32 = 128


        # # (bs * num_img, im_channels, h, w) ->
        # # (bs * num_img, im_channels, h / img_patch_strid, w / img_patch_strid) patches
        # ins = self.patchify(d0)
        # # (bs, im_channels, num_img, h / img_patch_strid, w / img_patch_strid) patches
        # ins = (
        #     ins.view(
        #         bs,
        #         num_img,
        #         self.im_channels,
        #         num_pat_img,
        #         num_pat_img,
        #     )
        #     .transpose(1, 2)
        #     .clone()
        # )

        # # concat proprio
        # _, _, _d, _h, _w = ins.shape
        # if self.add_proprio:
        #     p = self.proprio_preprocess(proprio)  # [B,4] -> [B,64]
        #     p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, _d, _h, _w)
        #     ins = torch.cat([ins, p], dim=1)  # [B, 128, num_img, np, np]

        # channel last
        ins = rearrange(ins, "b d ... -> b ... d")  # [B, num_img, np, np, 128]

        # save original shape of input for layer
        ins_orig_shape = ins.shape

        # flatten patches into sequence
        ins = rearrange(ins, "b ... d -> b (...) d")  # [B, num_img * np * np, 128]
        # add learable pos encoding
        # only added to image tokens
        if self.pe_fix:
            ins += self.pos_encoding

        # append language features as sequence
        num_lang_tok = 0
        if self.add_lang:
            l = self.lang_preprocess(
                lang_emb.view(bs * self.lang_max_seq_len, self.lang_emb_dim)
            )
            l = l.view(bs, self.lang_max_seq_len, -1)
            num_lang_tok = l.shape[1]
            ins = torch.cat((l, ins), dim=1)  # [B, num_img * np * np + 77, 128]

        # add learable pos encoding
        if not self.pe_fix:
            ins = ins + self.pos_encoding

        x = self.fc_bef_attn(ins)
        if self.self_cross_ver == 0:
            # self-attention layers
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

        elif self.self_cross_ver == 1:
            lx, imgx = x[:, :num_lang_tok], x[:, num_lang_tok:]

            # within image self attention
            imgx = imgx.reshape(bs * num_img, num_pat_img * num_pat_img, -1)
            for self_attn, self_ff in self.layers[: len(self.layers) // 2]:
                imgx = self_attn(imgx) + imgx
                imgx = self_ff(imgx) + imgx

            imgx = imgx.view(bs, num_img * num_pat_img * num_pat_img, -1)
            x = torch.cat((lx, imgx), dim=1)
            # cross attention
            for self_attn, self_ff in self.layers[len(self.layers) // 2 :]:
                x = self_attn(x) + x
                x = self_ff(x) + x

        else:
            assert False

        # append language features as sequence
        if self.add_lang:
            # throwing away the language embeddings
            x = x[:, num_lang_tok:]
        x = self.fc_aft_attn(x)

        # reshape back to orginal size
        x = x.view(bs, *ins_orig_shape[1:-1], x.shape[-1])  # [B, num_img, np, np, 128]
        x = rearrange(x, "b ... d -> b d ...")  # [B, 128, num_img, np, np]

        feat = []
        _feat = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]
        # _feat = _feat.view(bs, -1)
        # feat.append(_feat)
        _feat = rearrange(_feat, 'b c n -> b (c n)') # torch.Size([3, 640])
        feat.append(repeat(_feat, f'b d -> b {1} d'))  # torch.Size([3, ah, 640])

        x = (
            x.transpose(1, 2)
            .clone()
            .view(
                bs * self.num_img, self.input_dim_before_seq, num_pat_img, num_pat_img
            )
        )

        # if not train with memory, or this is fine branch
        if not self.use_memory or not self.no_feat:
            if self.cvx_up:
                # trick to stablize mixed-percision training
                with torch.cuda.amp.autocast(enabled=False):
                    conv1, ln1, act1, conv2, ln2, act2, up, conv3 = self.up0

                    feat_s0 = self.sam2_vision_feats_all[0].permute(1, 2, 0).reshape(bs*num_img, -1, num_pat_img*4, num_pat_img*4).float()
                    feat_s1 = self.sam2_vision_feats_all[1].permute(1, 2, 0).reshape(bs*num_img, -1, num_pat_img*2, num_pat_img*2).float()
                    x = x.float()
                    upscaled_embedding = act1(ln1(conv1(x) + feat_s1)) # [bs, 64, 32, 32]
                    upscaled_embedding = act2(ln2(conv2(upscaled_embedding) + feat_s0)) # [bs, 32, 64, 64]
                    trans = conv3(up(upscaled_embedding))

                trans = trans.view(bs, self.num_img, 1, h, w).half()
            else:
                u0 = self.up0(x)
                if self.inp_pre_con:
                    u0 = torch.cat([u0, d0], dim=1)
                u = self.final(u0)
                
                u = u.view(bs*num_img, 1, -1 ,h ,w)   #[3*5, ah, 64 , 220, 220 ]
                u = rearrange(u, 'bn ah c h w -> (bn ah) c h w') # torch.Size([3*5*ah, 64, 220, 220])

                # translation decoder
                trans = self.trans_decoder(u).view(bs, self.num_img, 1, h, w)

        # if train with memory, and this is corase branch
        else:

            self.vision_feats_all = [x.permute(2, 3, 0, 1).reshape(-1, *x.permute(2, 3, 0, 1).shape[2:]).clone()]
            self.vision_pos_embeds_all = [self.pos_encoding.expand(bs, *self.pos_encoding.shape).reshape(num_pat_img * num_pat_img, bs*self.num_img, 128).clone()]

            # training

            if self.training:
                num_obs = self.num_maskmem + 1
                num_seq = bs // num_obs
                bs_ = bs
                bs = 1
                role_label_seq = kwargs.get("role_label_seq")

                x_ = []
                u_ = []
                trans_ = []
                for seq_idx in range(num_seq):
                    self.reset_memory_bank()

                    for obs_idx in range(num_obs):

                        idx = num_obs * seq_idx + obs_idx
                        x_i = self.sam2_forward_with_memory(self.sam2, idx, num_img, feat_sizes)

                        x_.append(x_i)

                        if self.cvx_up:
                            with torch.cuda.amp.autocast(enabled=False):
                                conv1, ln1, act1, conv2, ln2, act2, up, conv3 = self.up0

                                feat_s0 = self.sam2_vision_feats_all[0].permute(1, 2, 0).reshape(bs_, num_img, -1, num_pat_img*4, num_pat_img*4)[idx].float()
                                feat_s1 = self.sam2_vision_feats_all[1].permute(1, 2, 0).reshape(bs_, num_img, -1, num_pat_img*2, num_pat_img*2)[idx].float()
                                x_i = x_i.float()
                                upscaled_embedding = act1(ln1(conv1(x_i) + feat_s1)) # [bs, 64, 32, 32]
                                upscaled_embedding = act2(ln2(conv2(upscaled_embedding) + feat_s0)) # [bs, 32, 64, 64]
                                trans = conv3(up(upscaled_embedding))

                            trans = trans.view(bs, self.num_img, 1, h, w).half()
                        else:
                            u0 = self.up0(x_i)
                            if self.inp_pre_con:
                                u0 = torch.cat([u0, d0], dim=1)
                            u = self.final(u0)
                            
                            u = u.view(bs*num_img, 1, -1 ,h ,w)   #[3*5, ah, 64 , 220, 220 ]
                            u = rearrange(u, 'bn ah c h w -> (bn ah) c h w') # torch.Size([3*5*ah, 64, 220, 220])

                            # translation decoder
                            trans = self.trans_decoder(u).view(bs, self.num_img, 1, h, w)

                            u_.append(u)

                        trans_.append(trans)

                        if self.use_memory:
                            curr_role_label = None
                            if role_label_seq is not None:
                                curr_role_label = role_label_seq[idx]
                            self.sam2_add_new_memory(
                                self.sam2,
                                trans.view(bs, num_img, 1, 1, h, w),
                                idx,
                                num_img,
                                feat_sizes,
                                role_label=curr_role_label,
                            )
                            self.curr_obs_idx += 1

                x = torch.cat(x_, dim=0)
                if not self.cvx_up:
                    u = torch.cat(u_, dim=0)
                trans = torch.cat(trans_, dim=0)
                bs = bs_


            # eval

            else:
                with torch.cuda.amp.autocast(enabled=True):
                    x = self.sam2_forward_with_memory(self.sam2, 0, num_img, feat_sizes)
                if self.cvx_up:
                    with torch.cuda.amp.autocast(enabled=False):
                        conv1, ln1, act1, conv2, ln2, act2, up, conv3 = self.up0

                        feat_s0 = self.sam2_vision_feats_all[0].permute(1, 2, 0).reshape(bs*num_img, -1, num_pat_img*4, num_pat_img*4).float()
                        feat_s1 = self.sam2_vision_feats_all[1].permute(1, 2, 0).reshape(bs*num_img, -1, num_pat_img*2, num_pat_img*2).float()
                        x = x.float()
                        upscaled_embedding = act1(ln1(conv1(x) + feat_s1)) # [bs, 64, 32, 32]
                        upscaled_embedding = act2(ln2(conv2(upscaled_embedding) + feat_s0)) # [bs, 32, 64, 64]
                        trans = conv3(up(upscaled_embedding))

                        # trans = self.up0(x)
                    trans = trans.view(bs, self.num_img, 1, h, w).half()
                else:
                    u0 = self.up0(x)
                    if self.inp_pre_con:
                        u0 = torch.cat([u0, d0], dim=1)
                    u = self.final(u0)
                    
                    u = u.view(bs*num_img, 1, -1 ,h ,w)   #[3*5, ah, 64 , 220, 220 ]
                    u = rearrange(u, 'bn ah c h w -> (bn ah) c h w') # torch.Size([3*5*ah, 64, 220, 220])

                    # translation decoder
                    trans = self.trans_decoder(u).view(bs, self.num_img, 1, h, w)

                if self.use_memory:
                    with torch.cuda.amp.autocast(enabled=True):
                        self.sam2_add_new_memory(
                            self.sam2,
                            trans.view(bs, num_img, 1, 1, h, w),
                            0,
                            num_img,
                            feat_sizes,
                        )
                    self.curr_obs_idx += 1
                
        if not self.no_feat:
            if self.feat_ver == 0:
                hm = F.softmax(trans.detach().view(bs, self.num_img, 1, h * w), 2).view(
                    bs * self.num_img * 1, 1, h, w
                )

                if self.cvx_up:
                    # since we donot predict u, we need to get u from x
                    # x is at a lower resolution than hm, therefore we average
                    # hm using the fold operation
                    _hm = F.unfold(
                        hm,
                        kernel_size=self.img_patch_size,
                        padding=0,
                        stride=self.img_patch_size,
                    )
                    assert _hm.shape == (
                        bs * self.num_img,
                        self.img_patch_size * self.img_patch_size,
                        num_pat_img * num_pat_img,
                    )
                    _hm = torch.mean(_hm, 1)
                    _hm = _hm.view(bs * self.num_img, 1, num_pat_img, num_pat_img)
                    _u = x
                else:
                    # (bs * num_img, self.input_dim_before_seq, h, w)
                    # we use the u directly
                    _hm = hm.view(bs * num_img * 1, 1, h, w)
                    _u = u

                _feat = torch.sum(_hm * _u, dim=[2, 3])
                # _feat = _feat.view(bs, -1)
                _feat = _feat.view(bs, num_img, 1, -1)  # torch.Size([3, 5, ah, 64])
                _feat = rearrange(_feat, 'b n ah c -> b ah (n c)')    # torch.Size([3, 5, ah, 64]) ->  torch.Size([3, ah, 5*64]) 
        

            elif self.feat_ver == 1:
                # get wpt_local while testing
                if not self.training:
                    wpt_local = self.get_wpt(
                        out={"trans": trans.clone().detach()},
                        dyn_cam_info=None,
                    )

                # projection
                # (bs, 1, num_img, 2)
                wpt_img = self.get_pt_loc_on_img(
                    wpt_local.unsqueeze(1),
                    dyn_cam_info=None,
                )
                wpt_img = wpt_img.reshape(bs * self.num_img, 2)

                # add noise to wpt image while training
                if self.training:
                    wpt_img = mvt_utils.add_uni_noi(
                        wpt_img, self.wpt_img_aug * self.img_size
                    )
                    wpt_img = torch.clamp(wpt_img, 0, self.img_size - 1)

                if self.cvx_up:
                    _wpt_img = wpt_img / self.img_patch_size
                    _u = x
                    assert (
                        0 <= _wpt_img.min() and _wpt_img.max() <= x.shape[-1]
                    ), print(_wpt_img, x.shape)
                else:
                    _u = u
                    _wpt_img = wpt_img

                _wpt_img = _wpt_img.unsqueeze(1)
                _feat = select_feat_from_hm(_wpt_img, _u)[0]
                _feat = _feat.view(bs, 1, -1)

            else:
                assert False, NotImplementedError

            feat.append(_feat)

            feat = torch.cat(feat, dim=-1)

            if self.rot_ver == 0:
                feat_norm = (feat - feat.mean()) / feat.std()
                feat = self.feat_fc(feat_norm)
                out = {"feat": feat}
                if self.graph_node_classes > 0:
                    out["graph_node_logits"] = self.graph_node_head(
                        feat_norm
                    ).unsqueeze(1)
            elif self.rot_ver == 1:
                feat = feat.squeeze(1)
                # features except rotation
                feat_ex_rot = self.feat_fc_ex_rot(feat)

                # batch normalized features for rotation
                feat_rot = self.feat_fc_init_bn(feat)
                feat_x = self.feat_fc_x(feat_rot)

                if self.training:
                    rot_x = rot_x_y[..., 0].view(bs, 1)
                else:
                    # sample with argmax
                    rot_x = feat_x.argmax(dim=1, keepdim=True)

                rot_x_pe = self.feat_fc_pe(rot_x)
                feat_y = self.feat_fc_y(feat_rot + rot_x_pe)

                if self.training:
                    rot_y = rot_x_y[..., 1].view(bs, 1)
                else:
                    rot_y = feat_y.argmax(dim=1, keepdim=True)
                rot_y_pe = self.feat_fc_pe(rot_y)
                feat_z = self.feat_fc_z(feat_rot + rot_x_pe + rot_y_pe)
                out = {
                    "feat_ex_rot": feat_ex_rot.unsqueeze(1),
                    "feat_x": feat_x.unsqueeze(1),
                    "feat_y": feat_y.unsqueeze(1),
                    "feat_z": feat_z.unsqueeze(1),
                }
                if self.graph_node_classes > 0:
                    out["graph_node_logits"] = self.graph_node_head(
                        feat
                    ).unsqueeze(1)
        else:
            out = {}

        if self.role_graph_enabled and self.role_graph_logits_runtime:
            out["role_graph_logits"] = torch.cat(
                self.role_graph_logits_runtime, dim=0
            ).unsqueeze(1)
        if self.phase_graph_logits_runtime:
            out["phase_graph_logits"] = torch.cat(
                self.phase_graph_logits_runtime, dim=0
            ).unsqueeze(1)
        if self.visit_mode_logits_runtime:
            out["visit_mode_logits"] = torch.cat(
                self.visit_mode_logits_runtime, dim=0
            ).view(-1, 1, 1)
        if self.role_ref_logits_runtime:
            out["role_ref_logits"] = torch.cat(
                self.role_ref_logits_runtime, dim=0
            ).unsqueeze(1)
        if self.anchor_use_logits_runtime:
            out["anchor_use_logits"] = torch.cat(
                self.anchor_use_logits_runtime, dim=0
            ).view(-1, 1, 1)
        if self.role_contrast_embeddings_runtime:
            out["role_contrast_embeddings"] = torch.cat(
                self.role_contrast_embeddings_runtime, dim=0
            )

        out.update({"trans": trans})

        return out

    def get_wpt(self, out, dyn_cam_info, y_q=None):
        """
        Estimate the q-values given output from mvt
        :param out: output from mvt
        """
        nc = self.num_img
        h = w = self.img_size
        bs = out["trans"].shape[0]

        q_trans = out["trans"].view(bs, nc, h * w)
        hm = torch.nn.functional.softmax(q_trans, 2)
        hm = hm.view(bs, nc, h, w)

        if dyn_cam_info is None:
            dyn_cam_info_itr = (None,) * bs
        else:
            dyn_cam_info_itr = dyn_cam_info

        pred_wpt = [
            self.renderer.get_max_3d_frm_hm_cube(
                hm[i : i + 1],
                fix_cam=True,
                dyn_cam_info=dyn_cam_info_itr[i : i + 1]
                if not (dyn_cam_info_itr[i] is None)
                else None,
            )
            for i in range(bs)
        ]
        pred_wpt = torch.cat(pred_wpt, 0)
        if self.use_point_renderer:
            pred_wpt = pred_wpt.squeeze(1)

        assert y_q is None

        return pred_wpt

    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        print("Freeing up some memory")
        self.renderer.free_mem()
