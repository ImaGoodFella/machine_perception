import torch.nn as nn

from src.nets.hand_heads.hand_hmr import HandHMR
from src.nets.hand_heads.mano_head import MANOHead
from src.nets.obj_heads.rigid_head import RigidHead

from src.nets.backbone.utils import get_backbone_info
import mp_lib.ld_utils as ld_utils
from mp_lib.unidict import xdict


class HMR(nn.Module):
    def __init__(self, backbone, focal_length, img_res, args):
        super(HMR, self).__init__()
        self.args = args
        if backbone == "resnet50":
            from src.nets.backbone.resnet import resnet50 as resnet
        elif backbone == "resnet18":
            from src.nets.backbone.resnet import resnet18 as resnet
        elif backbone == "resnet152":
            from src.nets.backbone.resnet import resnet152 as resnet
        elif backbone == 'resnext101_32x8d':
            from src.nets.backbone.resnet import resnext101_32x8d as resnet
        else:
            assert False

        self.backbone = resnet(pretrained=True)
        self.backbone_refine = resnet(pretrained=True)

        feat_dim = get_backbone_info(backbone)["n_output_channels"]
        self.head_r = HandHMR(feat_dim, is_rhand=True, n_iter=100)
        self.head_l = HandHMR(feat_dim, is_rhand=False, n_iter=100)

        self.mano_r = MANOHead(
            is_rhand=True, focal_length=focal_length, img_res=img_res
        )
        self.object_head = RigidHead(focal_length=focal_length, img_res=img_res)

        self.mode = "train"
        self.img_res = img_res
        self.focal_length = focal_length

    def inference(self, images, K):

        features = self.backbone(images)
        features_refine = self.backbone(features_refine)
        hmr_output_r = self.head_r(features, features_refine)

        # weak perspective
        root_r = hmr_output_r["cam_t.wp"]

        # forward hand
        mano_output_r = self.mano_r(
            rotmat=hmr_output_r["pose"],
            shape=hmr_output_r["shape"],
            K=K,
            cam=root_r,
        )

        out = {}
        out["j3d.cam.r"] = mano_output_r["j3d.cam.r"].cpu().detach()
        return out

    def forward(self, inputs, meta_info):
        images = inputs["img"]
        K = meta_info["intrinsics"]
        
        features = self.backbone(images)
        features_refine = self.backbone(features_refine)
        hmr_output_r = self.head_r(features, features_refine)

        # weak perspective
        root_r = hmr_output_r["cam_t.wp"]

        mano_output_r = self.mano_r(
            rotmat=hmr_output_r["pose"],
            shape=hmr_output_r["shape"],
            K=K,
            cam=root_r,
        )

        root_r_init = hmr_output_r["cam_t.wp.init"]
        mano_output_r.register("cam_t.wp.init.r", root_r_init)

        mano_output_r = ld_utils.prefix_dict(mano_output_r, "mano.")
        output = xdict()
        output.merge(mano_output_r)
        return output
