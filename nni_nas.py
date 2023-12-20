# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import sys
import pdb
import time
import torch
from pathlib import Path
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
from yacs.config import CfgNode

sys.path.append('.')
from fastreid.config import get_cfg
from fastreid.data import build_reid_test_loader, build_reid_train_loader
from fastreid.evaluation.testing import flatten_results_dict
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.modeling import build_model
from fastreid.solver import build_lr_scheduler, build_optimizer
from fastreid.evaluation import inference_on_dataset, print_csv_format, ReidEvaluator
from fastreid.utils.checkpoint import Checkpointer, PeriodicCheckpointer
from fastreid.utils import comm
from fastreid.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter
)

import nni
import nni.nas.strategy as strategy
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.experiment import NasExperiment
 
logger = logging.getLogger("fastreid")

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def evaluate_model_with_visualization(model, cfg):
    # dump the model into an onnx
    if 'NNI_OUTPUT_DIR' in os.environ:
        dummy_input = torch.zeros(1, 3, 32, 32).to(cfg.MODEL.DEVICE)
        torch.onnx.export(model, (dummy_input, ),
                          Path(os.environ['NNI_OUTPUT_DIR']) / 'model.onnx')

def get_evaluator(cfg, dataset_name, output_dir=None):
    data_loader, num_query = build_reid_test_loader(cfg, dataset_name=dataset_name)
    return data_loader, ReidEvaluator(cfg, num_query, output_dir)


def do_test(cfg, model):
    results = OrderedDict()
    for idx, dataset_name in enumerate(cfg.DATASETS.TESTS):
        logger.info("Prepare testing set")
        try:
            data_loader, evaluator = get_evaluator(cfg, dataset_name)
        except NotImplementedError:
            logger.warn(
                "No evaluator found. implement its `build_evaluator` method."
            )
            results[dataset_name] = {}
            continue
        results_i = inference_on_dataset(model, data_loader, evaluator, flip_test=cfg.TEST.FLIP.ENABLED)
        results[dataset_name] = results_i

        if comm.is_main_process():
            assert isinstance(
                results, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results
            )
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            results_i['dataset'] = dataset_name
            print_csv_format(results_i)

    return results


def do_train(model, cfg=None, resume=False):

    if isinstance(cfg, dict):
        cfg = CfgNode(cfg)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    data_loader = build_reid_train_loader(cfg)
    data_loader_iter = iter(data_loader)

    model.train()
    optimizer, param_wrapper= build_optimizer(cfg, model)

    iters_per_epoch = len(data_loader.dataset) // cfg.SOLVER.IMS_PER_BATCH
    scheduler = build_lr_scheduler(cfg, optimizer, iters_per_epoch)

    checkpointer = Checkpointer(
        model,
        cfg.OUTPUT_DIR,
        save_to_disk=comm.is_main_process(),
        optimizer=optimizer,
        **scheduler
    )

    start_epoch = 0 if not resume else 10
    iteration = start_iter = start_epoch * iters_per_epoch

    max_epoch = cfg.SOLVER.MAX_EPOCH
    max_iter = max_epoch * iters_per_epoch
    warmup_iters = cfg.SOLVER.WARMUP_ITERS
    delay_epochs = cfg.SOLVER.DELAY_EPOCHS

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_epoch)
    metric_name = cfg.DATASETS.TESTS[0] + "/metric"

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR)
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support some hooks, such as
    # accurate timing, FP16 training and precise BN here,
    # because they are not trivial to implement in a small training loop
    logger.info("Start training from epoch {}".format(start_epoch))
    with EventStorage(start_iter) as storage:
        for epoch in range(start_epoch, max_epoch):
            storage.epoch = epoch
            for _ in range(iters_per_epoch):
                data = next(data_loader_iter)
                storage.iter = iteration

                loss_dict = model(data)
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

                if iteration - start_iter > 5 and \
                        ((iteration + 1) % 200 == 0 or iteration == max_iter - 1) and \
                        ((iteration + 1) % iters_per_epoch != 0):
                    for writer in writers:
                        writer.write()

                iteration += 1

                if iteration <= warmup_iters:
                    scheduler["warmup_sched"].step()

            # Write metrics after each epoch
            for writer in writers:
                writer.write()

            if iteration > warmup_iters and (epoch + 1) > delay_epochs:
                scheduler["lr_sched"].step()

            if (cfg.TEST.EVAL_PERIOD > 0 
                and (epoch + 1) % cfg.TEST.EVAL_PERIOD == 0 
                and iteration != max_iter - 1) or (epoch + 1 == max_epoch):
                accuracy = 0
                results = do_test(cfg, model)
                print(results)
                for i in results.keys():
                    accuracy += results[i]['Rank-1']
                accuracy /= float(len(results))
                if epoch + 1 == max_epoch:
                    nni.report_final_result(accuracy)
                    evaluate_model_with_visualization(model, cfg)
                else:
                    nni.report_intermediate_result(accuracy)
            else:
                results = {}

            if results:
                flatten_results = flatten_results_dict(results)
                metric_dict = dict(metric=flatten_results[metric_name])
                periodic_checkpointer.step(epoch, **metric_dict)




def main(args):
    the_cfg = setup(args)
    model = build_model(the_cfg)
    logger.info("Model:\n{}".format(model))

    evaluator = FunctionalEvaluator(do_train, cfg=the_cfg, resume=args.resume)
    search_strategy = strategy.Random()  # dedup=False if deduplication is not wanted
    exp = NasExperiment(model, evaluator, search_strategy)
    exp.config.max_trial_number = 3   # spawn 3 trials at most
    exp.config.trial_concurrency = 1  # will run 1 trial concurrently
    exp.config.trial_gpu_number = 1   # 0 will not use GPU
    exp.config.training_service.use_active_gpu = True
    exp.run(port=8090) 
    for model_dict in exp.export_top_models(formatter='dict'):
        print(model_dict) 
    time.sleep(1e5)

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
