import torch
import mp_lib.data_utils as data_utils
from src.utils.mano import build_mano
from mp_lib.rend_utils import Renderer
from mp_lib.unidict import xdict
import mp_lib.ld_utils as ld_utils
from mp_lib.abstract_pl import AbstractPL
from mp_lib.comet_utils import push_images
from src.callbacks.eval_modules import eval_fn_dict


def mul_loss_dict(loss_dict):
    for key, val in loss_dict.items():
        loss, weight = val
        loss_dict[key] = loss * weight
    return loss_dict


class GenericWrapper(AbstractPL):
    def __init__(self, args):
        super().__init__(
            args,
            push_images,
            "loss__val",
            float("inf"),
            high_loss_val=float("inf"),
        )
        self.args = args
        self.mano_r = build_mano(is_rhand=True)
        self.add_module("mano_r", self.mano_r)
        self.renderer = Renderer(img_res=args.img_res)

    def set_flags(self, mode):
        self.model.mode = mode
        if mode == "train":
            self.train()
        else:
            self.eval()

    def forward(self, inputs, targets, meta_info, mode):

        models = {"mano_r": self.mano_r}

        models["object_head"] = self.model.object_head

        self.set_flags(mode)
        inputs = xdict(inputs)
        targets = xdict(targets)
        meta_info = xdict(meta_info)
        with torch.no_grad():
            inputs, targets, meta_info = self.process_fn(
                models, inputs, targets, meta_info, mode, self.args
            )

        move_keys = ["object.v_len"]
        for key in move_keys:
            if key in targets.keys():
                meta_info.register(key, targets[key])
        meta_info.register("mano.faces.r", self.mano_r.faces)
        pred = self.model(inputs, meta_info)
        loss_dict = self.loss_fn(
            pred=pred, gt=targets, meta_info=meta_info, args=self.args, epoch=self.trainer.current_epoch
        )
        loss_dict = {k: (loss_dict[k][0].mean(), loss_dict[k][1]) for k in loss_dict}
        loss_dict = mul_loss_dict(loss_dict)
        loss_dict["loss"] = sum(loss_dict[k] for k in loss_dict)

        # conversion for vis and eval
        keys = list(pred.keys())
        for key in keys:
            # denormalize 2d keypoints
            if "2d.norm" in key and key in pred.keys() and key in targets.keys():
                denorm_key = key.replace(".norm", "")

                val_pred = pred[key]
                val_gt = targets[key]

                val_denorm_pred = data_utils.unormalize_kp2d(
                    val_pred, self.args.img_res
                )
                val_denorm_gt = data_utils.unormalize_kp2d(val_gt, self.args.img_res)

                pred.register(denorm_key, val_denorm_pred)
                targets.register(denorm_key, val_denorm_gt)

        if mode == "train":
            return {"out_dict": (inputs, targets, meta_info, pred), "loss": loss_dict}

        if mode == "vis":
            vis_dict = xdict()
            vis_dict.merge(inputs.prefix("inputs."))
            vis_dict.merge(pred.prefix("pred."))
            vis_dict.merge(targets.prefix("targets."))
            vis_dict.merge(meta_info.prefix("meta_info."))
            vis_dict = vis_dict.detach()
            return vis_dict

        # evaluate metrics
        metrics_all = self.evaluate_metrics(
            pred, targets, meta_info, self.metric_dict
        ).to_torch()
        out_dict = xdict()
        out_dict.register("imgname", meta_info["imgname"])
        out_dict.merge(ld_utils.prefix_dict(metrics_all, "metric."))
        return out_dict, loss_dict

    def evaluate_metrics(self, pred, targets, meta_info, specs):
        metric_dict = xdict()
        for key in specs:
            metrics = eval_fn_dict[key](pred, targets, meta_info)
            metric_dict.merge(metrics)

        return metric_dict
