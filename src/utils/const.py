import sys
from src.utils.parser import construct_args
import mp_lib.comet_utils as comet_utils

parse_fn = construct_args
cmd = " ".join(sys.argv)
args = parse_fn()
args.cmd = cmd
experiment, args = comet_utils.init_experiment(args)
comet_utils.save_args(args, save_keys=["comet_key"])
