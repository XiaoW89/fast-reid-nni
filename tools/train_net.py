#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import nni
import sys, os
import pdb

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer

params = {
    'SOLVER.BASE_LR': 5e-3,
    'SOLVER.WEIGHT_DECAY': 5e-2,
    'MODEL.HEADS.POOL_LAYER': 'GlobalMaxPool',
    'SOLVER.IMS_PER_BATCH': 256,
    'SOLVER.MAX_EPOCH': 10,
    'SOLVER.OPT': 'AdamW',
    'MODEL.HEADS.MARGIN': 0.35,
    'MODEL.HEADS.SCALE': 64,
}

optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.nni_hpo:
        pid = os.getpid()
        nni_params = []
        for i in params: 
            nni_params.append(i)
            nni_params.append(params[i])
        nni_params += [
            'OUTPUT_DIR', cfg['OUTPUT_DIR'] + f"_{pid}"]
        cfg.merge_from_list(nni_params)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = DefaultTrainer.test(cfg, model)
        return res

    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
