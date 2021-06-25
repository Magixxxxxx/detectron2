from detectron2.evaluation.pascal_voc_evaluation import PascalVOCDetectionEvaluator
import os
import logging
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results, PascalVOCDetectionEvaluator
from detectron2.modeling import GeneralizedRCNNWithTTA

from myILOD.utils.register import register_dataset, my_register
from detectron2.evaluation.OWOD_evaluator import OWODEvaluator
class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        # return OWODEvaluator(dataset_name, cfg, distributed=True, output_dir=output_folder)
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

def main(args):

    my_register()
    cfg = setup(args)
    print(cfg)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    res = Trainer.test(cfg, model)
    return res

def setup(args):
    cfg = get_cfg()
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.merge_from_file(args.config_file)   # 从config file 覆盖配置
    cfg.merge_from_list(args.opts)          # 从CLI参数 覆盖配置

    dataset = "voc[20,20]" 
    model = "final"

    cfg.MODEL.WEIGHTS = "output/{}/model_{}.pth".format(dataset, model)
    cfg.OUTPUT_DIR = "./output/eval_{}_{}".format(dataset, model)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    args.config_file = "myILOD/configs/voc[20,20].yaml"
    args.num_gpus = 8
    args.dist_url = 'tcp://127.0.0.1:52124'
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )