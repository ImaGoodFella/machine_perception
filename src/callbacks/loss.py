import torch.nn as nn

from mp_lib.geometry import axis_angle_to_matrix
from src.callbacks.loss_modules import hand_kp3d_loss
from src.callbacks.loss_modules import vector_loss
from src.callbacks.loss_modules import joints_loss
from src.callbacks.loss_modules import mano_loss

mse_loss = nn.MSELoss(reduction="none")


def compute_loss(pred, gt, meta_info, args):
    # unpacking pred and gt
    pred_betas_r = pred["mano.beta.r"]
    pred_rotmat_r = pred["mano.pose.r"]
    pred_joints_r = pred["mano.j3d.cam.r"]
    pred_projected_keypoints_2d_r = pred["mano.j2d.norm.r"]

    gt_pose_r = gt["mano.pose.r"]
    gt_betas_r = gt["mano.beta.r"]
    gt_joints_r = gt["mano.j3d.cam.r"]
    gt_keypoints_2d_r = gt["mano.j2d.norm.r"]

    is_valid = gt["is_valid"]
    right_valid = gt["right_valid"]
    joints_valid_r = gt["joints_valid_r"]

    # reshape
    gt_pose_r = axis_angle_to_matrix(gt_pose_r.reshape(-1, 3)).reshape(-1, 16, 3, 3)

    # Compute loss on MANO parameters
    loss_regr_pose_r, loss_regr_betas_r = mano_loss(
        pred_rotmat_r,
        pred_betas_r,
        gt_pose_r,
        gt_betas_r,
        criterion=mse_loss,
        is_valid=right_valid,
    )

    # Compute 2D reprojection loss for the keypoints
    loss_keypoints_r = joints_loss(
        pred_projected_keypoints_2d_r,
        gt_keypoints_2d_r,
        criterion=mse_loss,
        jts_valid=joints_valid_r,
    )

    # Compute 3D keypoint loss
    loss_keypoints_3d_r = hand_kp3d_loss(
        pred_joints_r, gt_joints_r, mse_loss, joints_valid_r
    )

    # Compute camera translation loss
    loss_cam_t_r = vector_loss(
        pred["mano.cam_t.wp.r"],
        gt["mano.cam_t.wp.r"],
        mse_loss,
        right_valid,
    )
    loss_cam_t_r += vector_loss(
        pred["mano.cam_t.wp.init.r"],
        gt["mano.cam_t.wp.r"],
        mse_loss,
        right_valid,
    )

    # Return each loss and its weight contributin to the total loss
    # BIGBRAIN
    loss_dict = {
        "loss/mano/cam_t/r": (loss_cam_t_r, 0.01),
        "loss/mano/kp2d/r": (loss_keypoints_r, 0.001),
        "loss/mano/kp3d/r": (loss_keypoints_3d_r, 1.0),
        "loss/mano/pose/r": (loss_regr_pose_r, 0.001),
        "loss/mano/beta/r": (loss_regr_betas_r, 0.001),
    }
    return loss_dict
