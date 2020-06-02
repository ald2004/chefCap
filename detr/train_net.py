import logging
import os
import xml.etree.ElementTree as ET
from collections import OrderedDict
from functools import lru_cache

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
)
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.structures import BoxMode
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from fvcore.common.file_io import PathManager
from torch.nn.parallel import DistributedDataParallel
from models import mydetr

logger = logging.getLogger("detectron2_detr")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
things_class_dict = {
    'face-head': 0,
    'mask-head': 1,
    'face-cap': 2,
    'mask-cap': 3
}


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TRAIN = ("chefCap_train",)
    cfg.DATASETS.TEST = ("chefCap_val",)
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.NUM_GPUS = 2
    cfg.distributed = False
    cfg.device = "cuda"
    cfg.seed = 1314
    cfg.aux_loss = False
    cfg.backbone = 'resnet50'
    cfg.batch_size = 2
    cfg.bbox_loss_coef = 5
    cfg.clip_max_norm = 0.1
    cfg.coco_panoptic_path = 'None'
    cfg.coco_path = '/data/datasets/commondataset/coco'
    cfg.dataset_file = 'coco'
    cfg.dec_layers = 6
    cfg.dice_loss_coef = 1
    cfg.dilation = False
    cfg.dim_feedforward = 2048
    cfg.dist_url = 'env://'
    cfg.dropout = 0.1
    cfg.enc_layers = 6
    cfg.eos_coef = 0.1
    cfg.epochs = 300
    cfg.eval = True
    cfg.frozen_weights = 'None'
    cfg.giou_loss_coef = 2
    cfg.hidden_dim = 256
    cfg.lr = 0.0001
    cfg.lr_backbone = 1e-05
    cfg.lr_drop = 200
    cfg.mask_loss_coef = 1
    cfg.masks = False
    cfg.nheads = 8
    cfg.num_queries = 100
    cfg.num_workers = 2
    cfg.output_dir = 'output'
    cfg.position_embedding = 'sine'
    cfg.pre_norm = False
    cfg.remove_difficult = False
    cfg.resume = 'models/detr-r50-e632da11.pth'
    cfg.seed = 42
    cfg.set_cost_bbox = 5
    cfg.set_cost_class = 1
    cfg.set_cost_giou = 2
    cfg.start_epoch = 0
    cfg.weight_decay = 0.0001
    cfg.world_size = 1
    cfg.MODEL.WEIGHTS = "models/detr-r50-e632da11.pth"
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.MODEL.MASK_ON = False
    cfg.SOLVER.BASE_LR = 0.00025  # 学习率
    cfg.SOLVER.MAX_ITER = 10000  # 最大迭代次数 150000/8
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 只有一个类别：红绿灯
    cfg.NUM_GPUS = 2
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
        # record["image_id"] = idx
        record["image_id"] = infos['filename'].split('.')[0].strip()
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


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
                torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
                torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
            checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()

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
            scheduler.step()

            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and iteration % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def main(args):
    cfg = setup(args)
    for d in ["train", 'val']:
        # train for 6998images , val for 1199 images
        DatasetCatalog.register("chefCap_" + d, lambda d=d: get_chefcap_image_dicts("data/" + d))
        MetadataCatalog.get("chefCap_" + d).set(
            thing_classes=['face-head', 'mask-head', 'face-cap', 'mask-cap'])
        if d == 'val':
            MetadataCatalog.get("chefCap_val").evaluator_type = "pascal_voc"
            MetadataCatalog.get("chefCap_val").year = 2012
            MetadataCatalog.get("chefCap_val").dirname = "/opt/work/chefCap/data/val"

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    # if args.eval_only:
    #     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #         cfg.MODEL.WEIGHTS, resume=args.resume
    #     )
    #     return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    return do_train(cfg, model, resume=args.resume)
    # return do_test(cfg, model)


if __name__ == "__main__":
    # cd tools /
    # ./train_net.py --num-gpu 2 --resume --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
    # python train_net.py --num-gpu 2 --resume --config-file cfg/chef_cap.yaml
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
