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
    'LR': 5e-3,
    'WD': 5e-2,
    'PoolType': 'GlobalMaxPool',
    'BatchSize': 256,
    'MaxEpoch': 10,
    'Optimizer': 'AdamW',
    'HeadMargin': 0.35,
    'HeadScale': 64,
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
        nni_param = [
                'SOLVER.BASE_LR', params['LR'], 
                'SOLVER.WEIGHT_DECAY', params['WD'],
                'SOLVER.IMS_PER_BATCH', params['BatchSize'],
                'MODEL.HEADS.POOL_LAYER', params['PoolType'],
                'SOLVER.MAX_EPOCH', params['MaxEpoch'],
                'SOLVER.OPT', params['Optimizer'],
                'MODEL.HEADS.MARGIN', params['HeadMargin'],
                'MODEL.HEADS.SCALE', params['HeadScale'],
                'OUTPUT_DIR', cfg['OUTPUT_DIR'] + f"_{pid}", 
                ]
        cfg.merge_from_list(nni_param)
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
