import itertools
import logging
import os
import matplotlib

matplotlib.use('Agg')
from collections import (
    OrderedDict,
)
import glob
import detectron2.utils.comm as comm
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    DatasetFromList,
)
from detectron2.data.common import MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers.distributed_sampler import InferenceSampler
from detectron2.engine import (
    default_argument_parser,
    launch,
    default_setup,
    DefaultTrainer,
    hooks,
)
from detectron2.evaluation.cityscapes_evaluation import CityscapesEvaluator
from detectron2.evaluation import (
    COCOEvaluator,
    verify_results,
    pascal_voc_evaluation,
    PascalVOCDetectionEvaluator,
    # CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.structures import BoxMode
from detectron2.utils.logger import log_first_n
from tabulate import tabulate
from termcolor import colored
from evaluator import parse_rec as parse_rec

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
things_class_dict = {
    'face-head': 0,
    'mask-head': 1,
    'face-cap': 2,
    'mask-cap': 3,
    'uniform': 4,
    'non-uniform': 5
}


def print_instances_class_histogram(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    for entry in dataset_dicts:
        annos = entry["annotations"]
        classes = [x["category_id"] for x in annos if not x.get("iscrowd", 0)]
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    log_first_n(
        logging.INFO,
        "Distribution of instances among all {} categories:\n".format(num_classes)
        + colored(table, "cyan"),
        key="message",
    )


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TRAIN = ("chefCap_train",)
    cfg.DATASETS.TEST = ("chefCap_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    # cfg.MODEL.WEIGHTS = "model/model_final_68b088.pkl"

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    cfg.MODEL.MASK_ON = False
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 10000
    # cfg.NUM_GPUS = 2
    # cfg.INPUT.MIN_SIZE_TRAIN=800
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(things_class_dict.keys())
    cfg.TEST.AUG.ENABLED = False
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    modelkey = ["model_final_68b088.pkl"]
    modelmAP50_val = [20.1884, 28.9375, 26.6818, 27.6115, 20.1884, 19.7896, 10.8858]
    cfg.MODEL.WEIGHTS = f"model/{modelkey[0]}"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # or 0.7
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def get_chefcap_image_dicts(img_dir='data'):
    data_dirs = glob.glob(os.path.join(img_dir) + "/*")
    file_ext = '.xml'
    dataset_dicts = []
    idx = 0
    for data_dir in data_dirs:
        file_ids = glob.glob(os.path.join(data_dir, 'Annotations') + f'/*{file_ext}')
        # with open(os.path.join(img_dir, 'file_id.txt'), 'r') as fid:
        #     file_ids = list(map(str.strip, fid.readlines()))
        # assert file_ids
        for filename in file_ids:
            record = {}
            # filename = os.path.join(img_dir, 'Annotations', v) + file_ext
            infos = parse_rec(filename)
            # height, width = detection_utils.read_image(filename, format="BGR")
            record["file_name"] = os.path.join(data_dir, 'JPEGImages', infos['filename'])
            record["image_id"] = idx
            # record["image_id"] = infos['filename'].split('.')[0].strip()
            # 'size': {'width': '1280', 'height': '720', 'depth': '3'},
            record["height"] = int(infos["size"]["height"])
            record["width"] = int(infos["size"]["width"])
            record["segmented"] = 0  # infos["segmented"]
            annos = infos["objects"]
            objs = []
            for anno in annos:
                assert not anno["truncated"]
                obj = {
                    "bbox": anno['bbox'],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [],
                    "category_id": things_class_dict[anno["name"]],
                    "iscrowd": 0,
                    "difficult": anno["difficult"]
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts


def main(args):
    # if args.unitest:
    #     return unitest()
    cfg = setup(args)

    for d in ["train", 'val']:
        # train for 6998images , val for 1199 images
        DatasetCatalog.register("chefCap_" + d, lambda d=d: get_chefcap_image_dicts())
        MetadataCatalog.get("chefCap_" + d).set(
            thing_classes=list(things_class_dict.keys()))
        if d == 'val':
            MetadataCatalog.get("chefCap_val").evaluator_type = "pascal_voc"
            MetadataCatalog.get("chefCap_val").year = 2012
            MetadataCatalog.get("chefCap_val").dirname = "/opt/work/chefCap/detectron2_fasterrcnn/data"

    # for d in ["/opt/work/chefCap/data/ziped/Making-PascalVOC-export/"]:
    #     DatasetCatalog.register("chefCap_val",
    #                             lambda d=d: get_chefcap_image_dicts(d))
    # MetadataCatalog.get("chefCap_val").set(
    #     thing_classes=['face-head', 'mask-head', 'face-cap', 'mask-cap'])
    # MetadataCatalog.get("chefCap_val").evaluator_type = "pascal_voc"
    # MetadataCatalog.get("chefCap_val").dirname = "/opt/work/chefCap/data/ziped/Making-PascalVOC-export/"
    # MetadataCatalog.get("chefCap_val").year = 2012
    if args.eval_only:
        model = DefaultTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = DefaultTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(DefaultTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    # if cfg.TEST.AUG.ENABLED:
    #     trainer.register_hooks(
    #         [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
    #     )
    return trainer.train()


def unitest():
    import glob
    for f in glob.glob("/opt/work/chefCap/data/train/Annotations/*"):
        k = parse_rec(f)
        print(k['size'])
    return 0


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    args.config_file = 'detectron2_cfg/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
    print(f'Command line Args: {args}')
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url="auto",
        args=(args,),
    )
    # python train_net.py --num-gpus 1 --resume
# |  category  | #instances   |  category  | #instances   |  category   | #instances   |
# |:----------:|:-------------|:----------:|:-------------|:-----------:|:-------------|
# | face-head  | 2464         | mask-head  | 4364         |  face-cap   | 624          |
# |  mask-cap  | 13622        |  uniform   | 11814        | non-uniform | 2098         |
# |            |              |            |              |             |              |
# |   total    | 34986        |            |              |             |              |
