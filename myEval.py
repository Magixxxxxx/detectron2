import os
import logging
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.modeling import GeneralizedRCNNWithTTA


# 数据集路径
DATASET_ROOT = '/root/userfolder/data/voc2007'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'train2007')
VAL_PATH = os.path.join(DATASET_ROOT, 'val2007')
TRAIN_JSON = os.path.join(ANN_ROOT, '[1, 20]-voc_train2007.json')
VAL_JSON = os.path.join(ANN_ROOT, '[1, 20]-voc_val2007.json')


# 数据集类别元数据
DATASET_CATEGORIES = {
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 
    6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 
    11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 
    16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
}


# 数据集的子集
PREDEFINED_SPLITS_DATASET = {
    "train_voc2007": (TRAIN_PATH, TRAIN_JSON),
    "val_voc2007": (VAL_PATH, VAL_JSON),
}


def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key, 
                                   metadate=get_dataset_instances_meta(), 
                                   json_file=json_file, 
                                   image_root=image_root)

def get_dataset_instances_meta():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k for k,_ in DATASET_CATEGORIES.items()]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [v for _,v in DATASET_CATEGORIES.items()]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret

def register_dataset_instances(name, metadate, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file, 
                                  image_root=image_root, 
                                  evaluator_type="coco", 
                                  **metadate)


# 注册数据集和元数据
# def plain_register_dataset():
#     DatasetCatalog.register("train_voc2007", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH, "train_voc2007"))
#     MetadataCatalog.get("train_voc2007").set(thing_classes=["pos", "neg"],
#                                                     json_file=TRAIN_JSON,
#                                                     image_root=TRAIN_PATH)
#     DatasetCatalog.register("val_voc2007", lambda: load_coco_json(VAL_JSON, VAL_PATH, "val_voc2007"))
#     MetadataCatalog.get("val_voc2007").set(thing_classes=["pos", "neg"],
#                                                 json_file=VAL_JSON,
#                                                 image_root=VAL_PATH)

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_folder)

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
    cfg = get_cfg()
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.merge_from_file(args.config_file)   # 从config file 覆盖配置
    cfg.merge_from_list(args.opts)          # 从CLI参数 覆盖配置

    cfg.MODEL.WEIGHTS = "TEST/model_final.pth"
    cfg.OUTPUT_DIR = "./output/eval"
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    print(cfg)
    
    # 注册数据集
    register_dataset()

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    res = Trainer.test(cfg, model)
    return res

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    args.config_file = "myILOD/Base-RCNN-C4.yaml"
    args.num_gpus = 4
    args.dist_url = 'tcp://127.0.0.1:52124'
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )