from src.models.generic.wrapper import GenericWrapper
from src.models.hmr.model import HMR
from src.callbacks.visualize import visualize_all
from src.callbacks.process import process_data
from src.callbacks.loss import compute_loss


class Wrapper(GenericWrapper):
    def __init__(self, args):
        super().__init__(args)
        self.model = HMR(
            backbone='resnext101_32x8d',
            #backbone="resnet152",
            #backbone="resnet18",
            focal_length=args.focal_length,
            img_res=args.img_res,
            args=args,
        )
        self.process_fn = process_data
        self.loss_fn = compute_loss
        self.metric_dict = [
            "mpjpe.ra",
        ]

        self.vis_fns = [visualize_all]

        self.num_vis_train = 1
        self.num_vis_val = 1
