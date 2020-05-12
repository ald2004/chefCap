import logging
import os
import xml.etree.ElementTree as ET
from collections import OrderedDict
from functools import lru_cache

from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    # detection_utils,
)
from detectron2.engine import (
    default_argument_parser,
    launch,
    default_setup,
    DefaultTrainer,
    hooks,
)
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
things_class_dict = {
    'face-head': 0,
    'mask-head': 1,
    'face-cap': 2,
    'mask-cap': 3
}


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super(Trainer, self).__init__(cfg=cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, 'evaluation')
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger('detectron2.trainer')
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info('Running inference with test-time augmentation ...')
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, 'inference_TTA'))
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + '_TTA': v for k, v in res.items()})
        return res


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TRAIN = ("chefCap_train",)
    cfg.DATASETS.TEST = ("chefCap_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "model/model_final_68b088.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    cfg.MODEL.MASK_ON = False
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 10000
    cfg.NUM_GPUS = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    # cfg.INPUT.MIN_SIZE_TRAIN=800
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(things_class_dict.keys())
    cfg.TEST.AUG.ENABLED = True
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


@lru_cache(maxsize=None)
def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    with PathManager.open(filename) as f:
        tree = ET.parse(f)

    infos = {}
    infos["folder"] = tree.find("folder").text.strip()
    infos["filename"] = tree.find("filename").text.strip()
    infos["path"] = tree.find("path").text.strip()
    infos["source"] = tree.find("source").find("database").text.strip()
    infos["size"] = {
        "width": tree.find("size").find("width").text.strip(),
        "height": tree.find("size").find("height").text.strip(),
        "depth": tree.find("size").find("depth").text.strip()
    }
    infos["segmented"] = tree.find("segmented").text.strip()
    objects = []
    infos["objects"] = objects
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        try:
            obj_struct["bbox"] = [
                int(bbox.find("xmin").text),
                int(bbox.find("ymin").text),
                int(bbox.find("xmax").text),
                int(bbox.find("ymax").text),
            ]
        except ValueError:
            obj_struct["bbox"] = [
                int(float(bbox.find("xmin").text)),
                int(float(bbox.find("ymin").text)),
                int(float(bbox.find("xmax").text)),
                int(float(bbox.find("ymax").text)),
            ]
        objects.append(obj_struct)
    return infos


def get_chefcap_image_dicts(img_dir='train'):
    with open(os.path.join(img_dir, 'file_id.txt'), 'r') as fid:
        file_ids = list(map(str.strip, fid.readlines()))
    assert file_ids
    file_ext = '.xml'
    dataset_dicts = []
    for idx, v in enumerate(file_ids):
        record = {}
        filename = os.path.join(img_dir, 'Annotations', v) + file_ext
        infos = parse_rec(filename)
        # height, width = detection_utils.read_image(filename, format="BGR")
        record["file_name"] = os.path.join(img_dir, 'JPEGImages', infos['filename'])
        record["image_id"] = idx
        # 'size': {'width': '1280', 'height': '720', 'depth': '3'},
        record["height"] = int(infos["size"]["height"])
        record["width"] = int(infos["size"]["width"])
        record["segmented"] = infos["segmented"]
        annos = infos["objects"]
        objs = []
        for anno in annos:
            assert not anno["truncated"]
            difficult = anno["difficult"]
            obj = {
                "bbox": anno['bbox'],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [],
                "category_id": things_class_dict[anno["name"]],
                "iscrowd": 0,
                "difficult": difficult
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def main(args):
    if args.unitest:
        return unitest()
    cfg = setup(args)
    if args.eval_only:
        return 0
    for d in ["train", "val"]:
        # train for 6998images , val for 1199 images
        DatasetCatalog.register("chefCap_" + d, lambda d=d: get_chefcap_image_dicts("data/" + d))
        MetadataCatalog.get("chefCap_" + d).set(
            thing_classes=['face-head', 'mask-head', 'face-cap', 'mask-cap'])
        if d == 'val':
            MetadataCatalog.get("chefCap_val").evaluator_type = "coco"

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


def unitest():
    import glob
    for f in glob.glob("/opt/work/chefCap/data/train/Annotations/*"):
        k = parse_rec(f)
        print(k['size'])
    return 0


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    args.config_file = 'cfg/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
    args.num_gpus = 2
    print(f'Command line Args: {args}')
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url="auto",
        args=(args,),
    )
# |  category  | #instances   |  category  | #instances   |  category  | #instances   |
# |:----------:|:-------------|:----------:|:-------------|:----------:|:-------------|
# | face-head  | 213          | mask-head  | 1397         |  face-cap  | 337          |
# |  mask-cap  | 4955         |            |              |            |              |
# |   total    | 6902         |            |              |            |              |
