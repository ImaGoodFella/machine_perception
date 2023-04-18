import mp_lib.metrics as metrics


def eval_mpjpe_ra(pred, targets, meta_info):
    joints3d_cam_r_gt = targets["mano.j3d.cam.r"]
    joints3d_cam_r_pred = pred["mano.j3d.cam.r"]
    is_valid = targets["is_valid"]
    right_valid = targets["right_valid"] * is_valid

    joints3d_cam_r_gt_ra = joints3d_cam_r_gt - joints3d_cam_r_gt[:, :1, :]
    joints3d_cam_r_pred_ra = joints3d_cam_r_pred - joints3d_cam_r_pred[:, :1, :]
    mpjpe_ra_r = metrics.compute_joint3d_error(
        joints3d_cam_r_gt_ra, joints3d_cam_r_pred_ra, right_valid
    )
    metric_dict = {}
    metric_dict["mpjpe/ra/r"] = mpjpe_ra_r.mean(axis=1) * 1000
    return metric_dict


def eval_mpvpe_ra(pred, targets, meta_info):
    root_r_gt = targets["mano.j3d.cam.r"][:, :1]
    root_r_pred = pred["mano.j3d.cam.r"][:, :1]

    v3d_cam_r_gt = targets["mano.v3d.cam.r"]
    v3d_cam_r_pred = pred["mano.v3d.cam.r"]
    is_valid = targets["is_valid"]
    right_valid = targets["right_valid"] * is_valid

    v3d_cam_r_gt_ra = v3d_cam_r_gt - root_r_gt
    v3d_cam_r_pred_ra = v3d_cam_r_pred - root_r_pred
    mpjpe_ra_r = metrics.compute_joint3d_error(
        v3d_cam_r_gt_ra, v3d_cam_r_pred_ra, right_valid
    )
    metric_dict = {}
    metric_dict["mpvpe/ra/r"] = mpjpe_ra_r.mean(axis=1) * 1000
    return metric_dict


eval_fn_dict = {
    "mpjpe.ra": eval_mpjpe_ra,
    "mpvpe.ra": eval_mpvpe_ra,
}
