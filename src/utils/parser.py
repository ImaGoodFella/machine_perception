import argparse
from easydict import EasyDict


def construct_args():
    """
    Generic options that are non-specific to a project.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_ckpt", type=str, default="", help="Load checkpoints from PL format"
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default="",
        help="Resume training from checkpoint and keep logging in the same comet exp",
    )
    parser.add_argument(
        "-f",
        "--fast",
        dest="fast_dev_run",
        help="dry run mode with smaller number of iterations",
        action="store_true",
    )
    parser.add_argument(
        "--trainsplit",
        type=str,
        default="train",
        choices=["train", "smalltrain", "minitrain", "tinytrain", "fulltrain"],
        help="Amount to subsample training set.",
    )
    parser.add_argument(
        "--valsplit",
        type=str,
        default="tinyval",
        choices=["val", "smallval", "tinyval", "minival", "none"],
        help="Amount to subsample validation set.",
    )

    parser.add_argument("--log_every", type=int, default=50, help="log every k steps")
    parser.add_argument(
        "--eval_every_epoch", type=int, default=4, help="Eval every k epochs"
    )
    parser.add_argument(
        "--lr_dec_epoch",
        type=int,
        nargs="+",
        default=[],
        help="Learning rate decay epoch.",
    )
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--lr_decay", type=float, default=0.1, help="Learning rate decay factor"
    )
    parser.add_argument("--acc_grad", type=int, default=1, help="Accumulate gradient")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--mute", help="No logging with comet", action="store_true")
    parser.add_argument("--no_vis", help="Stop visualization", action="store_true")
    parser.add_argument("--exp_key", type=str, default="")
    args = parser.parse_args()
    args = EasyDict(vars(parser.parse_args()))

    args.focal_length = 1000.0
    args.img_res = 224

    # Data augmentation
    args.rot_factor = 10.0
    args.noise_factor = 0.1
    args.scale_factor = 0.1
    args.flip_prob = 0.0

    args.img_norm_mean = [0.485, 0.456, 0.406]
    args.img_norm_std = [0.229, 0.224, 0.225]
    args.pin_memory = True
    args.shuffle_train = True
    args.seed = 1
    args.grad_clip = 150.0

    if args.fast_dev_run:
        args.num_workers = 0
        args.batch_size = 8
        args.trainsplit = "minitrain"
        args.valsplit = "minival"
        args.log_every = 1
        args.eval_every_epoch = 1
    return args
