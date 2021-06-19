import os
import logging
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results, PascalVOCDetectionEvaluator
from detectron2.modeling import GeneralizedRCNNWithTTA

from myILOD.utils.register import register_dataset, my_register

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return PascalVOCDetectionEvaluator(dataset_name)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg() # 拷贝default config副本

    # 更改配置参数
    # cfg.DATASETS.TRAIN = ("train_voc2007",)
    # cfg.DATASETS.TEST = ("val_voc2007",)
    # cfg.INPUT.MAX_SIZE_TRAIN = 400
    # cfg.INPUT.MAX_SIZE_TEST = 400
    # cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    # cfg.INPUT.MIN_SIZE_TEST = 160
    # ITERS_IN_ONE_EPOCH = int(5011 / cfg.SOLVER.IMS_PER_BATCH) # iters_in_one_epoch = dataset_imgs/batch_size  
    # cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 12) - 1 # 12 epochs
    # cfg.SOLVER.BASE_LR = 0.02
    # cfg.SOLVER.MOMENTUM = 0.9
    # cfg.SOLVER.WEIGHT_DECAY = 0.0001
    # cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    # cfg.SOLVER.GAMMA = 0.1
    # cfg.SOLVER.STEPS = (30000,)
    # cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    # cfg.SOLVER.WARMUP_ITERS = 100
    # cfg.SOLVER.WARMUP_METHOD = "linear"
    # cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1

    cfg.merge_from_file(args.config_file)   # 从config file 覆盖配置
    cfg.merge_from_list(args.opts)          # 从CLI参数 覆盖配置
    cfg.freeze()
    default_setup(cfg, args)

    return cfg

def main(args):
    
    my_register()
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.config_file = "myILOD/configs/voc[1,10].yaml"
    args.num_gpus = 4
    args.dist_url='tcp://127.0.0.1:52133'
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )