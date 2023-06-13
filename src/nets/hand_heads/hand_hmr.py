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

        self.cam_init = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 3),
        )

        self.shape_init = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 10),
        )

        self.pose_init = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 16 * 3),
        )
        
        self.hand_specs = hand_specs
        self.n_iter = n_iter
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def init_vector_dict(self, feat_pose, feat_shape, feat_cam):
        batch_size = feat_pose.shape[0]
        dev = feat_pose.device
        init_pose = (
            matrix_to_rotation_6d(axis_angle_to_matrix(self.pose_init(feat_pose).reshape(batch_size, 16, 3)))
            .reshape(batch_size, -1)
        )

        init_shape = self.shape_init(feat_shape)
        init_transl = self.cam_init(feat_cam)

        out = {}
        out["pose_6d"] = init_pose
        out["shape"] = init_shape
        out["cam_t/wp"] = init_transl
        out = xdict(out).to(dev)

        return out

    def forward(self, features_pose, features_shape, features_cam, features_refine, use_pool=True):

        batch_size = features_pose.shape[0]
        if use_pool:
            feat_pose = self.avgpool(features_pose)
            feat_shape = self.avgpool(features_shape)
            feat_cam = self.avgpool(features_cam)
            feat_refine = self.avgpool(features_refine)

            feat_pose = feat_pose.view(feat_pose.size(0), -1)
            feat_shape = feat_shape.view(feat_shape.size(0), -1)
            feat_cam = feat_cam.view(feat_cam.size(0), -1)
            feat_refine = feat_refine.view(feat_refine.size(0), -1)
        else:
            feat_pose = features_pose
            feat_shape = features_shape
            feat_cam = features_cam
            feat_refine = features_refine

        init_vdict = self.init_vector_dict(feat_pose, feat_shape, feat_cam)
        init_cam_t = init_vdict["cam_t/wp"].clone()
        pred_vdict = self.hmr_layer(feat_refine, init_vdict, self.n_iter)

        pred_rotmat = rotation_6d_to_matrix(pred_vdict["pose_6d"].reshape(-1, 6)).view(
            batch_size, 16, 3, 3
        )

        pred_vdict.register("pose", pred_rotmat)
        pred_vdict.register("cam_t.wp.init", init_cam_t)
        pred_vdict = pred_vdict.replace_keys("/", ".")
        return pred_vdict
