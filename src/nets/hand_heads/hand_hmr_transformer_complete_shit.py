import torch
import torch.nn as nn
from src.nets.hmr_layer import HMRLayer
from mp_lib.unidict import xdict
from mp_lib.geometry import (
    matrix_to_rotation_6d,
    axis_angle_to_matrix,
    rotation_6d_to_matrix,
)

class HandHMR(nn.Module):
    def __init__(self, feat_dim, is_rhand, n_iter):
        super().__init__()
        self.is_rhand = is_rhand

        hand_specs = {"pose_6d": 6 * 16, "cam_t/wp": 3, "shape": 10}
        self.hmr_layer = HMRLayer(feat_dim, 1024, hand_specs)

        self.pose_init = PoseInitTransformer(feat_dim, hand_specs["pose_6d"])
        self.cam_init = CamInitTransformer(feat_dim, hand_specs["cam_t/wp"])
        self.shape_init = ShapeInitTransformer(feat_dim, hand_specs["shape"])

        self.hand_specs = hand_specs
        self.n_iter = n_iter
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def init_vector_dict(self, features):
        batch_size = features.shape[0]
        dev = features.device
        init_pose = self.pose_init(features)
        init_shape = self.shape_init(features)
        init_transl = self.cam_init(features)

        out = {}
        out["pose_6d"] = init_pose
        out["shape"] = init_shape
        out["cam_t/wp"] = init_transl
        out = xdict(out).to(dev)
        return out

    def forward(self, features, use_pool=True):
        batch_size = features.shape[0]
        if use_pool:
            feat = self.avgpool(features)
            feat = feat.view(feat.size(0), -1)
        else:
            feat = features

        init_vdict = self.init_vector_dict(feat)
        init_cam_t = init_vdict["cam_t/wp"].clone()

        # Pose initialization
        init_pose = self.pose_init(feat)
        pred_vdict = init_vdict
        pred_vdict["pose_6d"] = init_pose

        for i in range(self.n_iter):
            pred_vdict = self.hmr_layer(feat, pred_vdict, 1)
            pred_vdict["pose_6d"] = pred_vdict["pose_6d"] + init_pose

        # Forward prediction of pose
        pred_rotmat = rotation_6d_to_matrix(pred_vdict["pose_6d"].reshape(-1, 6)).view(
            batch_size, 16, 3, 3
        )
        pred_vdict.register("pose", pred_rotmat)
        pred_vdict.register("cam_t.wp.init", init_cam_t)
        pred_vdict = pred_vdict.replace_keys("/", ".")

        return pred_vdict


class PoseInitTransformer(nn.Module):
    def __init__(self, feat_dim, output_dim):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feat_dim, nhead=8),
            num_layers=6
        )
        self.linear = nn.Linear(feat_dim, output_dim)

    def forward(self, features):
        transformer_out = self.transformer(features.unsqueeze(1))
        pose_init = self.linear(transformer_out.squeeze(1))
        return pose_init


class CamInitTransformer(nn.Module):
    def __init__(self, feat_dim, output_dim):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feat_dim, nhead=8),
            num_layers=4
        )
        self.linear = nn.Linear(feat_dim, output_dim)

    def forward(self, features):
        transformer_out = self.transformer(features.unsqueeze(1))
        cam_init = self.linear(transformer_out.squeeze(1))
        return cam_init


class ShapeInitTransformer(nn.Module):
    def __init__(self, feat_dim, output_dim):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feat_dim, nhead=8),
            num_layers=4
        )
        self.linear = nn.Linear(feat_dim, output_dim)

    def forward(self, features):
        transformer_out = self.transformer(features.unsqueeze(1))
        shape_init = self.linear(transformer_out.squeeze(1))
        return shape_init