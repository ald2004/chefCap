# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import base64
import functools
# from utils import setup_logger
# from utils import YOLO_single_img
import logging
import os
import sys
import time
import warnings

import cv2
import numpy as np
# tf
# pytorch
from detectron2.utils.visualizer import Visualizer
from fvcore.common.file_io import PathManager
from termcolor import colored
import matplotlib.colors as mplc
import darknet

try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
    ImageEnhance = None

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS
_SMALL_OBJECT_AREA_THRESH = 1000
HAS_GPU = False
darknet.hasGPU = HAS_GPU
if HAS_GPU:
    from darknet import set_gpu


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return PathManager.open(filename, "a")


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(
        output=None, distributed_rank=0, *, color=True, name="chefCap", abbrev_name=None, log_level=logging.DEBUG
):
    """
    Initialize the detectron2 logger and set its verbosity level to "DEBUG".

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        abbrev_name (str): an abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.
            By default, will abbreviate "detectron2" to "d2" and leave other
            modules unchanged.

    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = "d2" if name == "detectron2" else name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(threadName)-9s %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(threadName)-9s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)
        PathManager.mkdirs(os.path.dirname(filename))

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


def load_img(path, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest'):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file.
        grayscale: DEPRECATED use `color_mode="grayscale"`.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if grayscale is True:
        warnings.warn('grayscale is deprecated. Please use '
                      'color_mode = "grayscale"')
        color_mode = 'grayscale'
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `load_img` requires PIL.')
    img = pil_image.open(path)
    if color_mode == 'grayscale':
        if img.mode != 'L':
            img = img.convert('L')
    elif color_mode == 'rgba':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    elif color_mode == 'rgb':
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img


def img_to_array_raw(img, data_format='channels_last', dtype='float32'):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: %s' % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (x.shape,))
    return x


class myx_Visualizer(Visualizer):
    def overlay_instances(
            self,
            *,
            boxes=None,
            labels=None,
            masks=None,
            keypoints=None,
            assigned_colors=None,
            alpha=0.5
    ):

        num_instances = None
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if keypoints is not None:
            if num_instances:
                assert len(keypoints) == num_instances
            else:
                num_instances = len(keypoints)
            keypoints = self._convert_keypoints(keypoints)
        if labels is not None:
            assert len(labels) == num_instances

        if num_instances == 0:
            return self.output
        if boxes is not None and boxes.shape[1] == 5:
            return self.overlay_rotated_instances(
                boxes=boxes, labels=labels, assigned_colors=assigned_colors
            )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])
        # assigned_colors = np.array([  # RGB
        #     # [0.667, 0.333, 1.],
        #     # [0.85, 0.325, 0.098],
        #     # [0., 0.667, 0.5],
        #     # [0.749, 0.749, 0.],
        #     # [0.333, 0.667, 0.000],
        #     # [0.000, 0.667, 0.500]
        #     [0.333, 0.667, 0.000],
        #     [0.92941176, 0.02745098, 0.05098039],
        #     [0.92941176, 0.02745098, 0.05098039],
        #     [0.92941176, 0.02745098, 0.05098039],
        #     [0.333, 0.667, 0.000],
        #     [0.333, 0.667, 0.000],
        #     [0.92941176, 0.02745098, 0.05098039]
        # ], dtype=np.float)
        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            # assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None
        COLOR_RED = np.array([0.05098039, 0.02745098, 0.92941176])
        COLOR_GREED = np.array([0.000, 0.767, 0.133])
        color_dic = {'face-head': COLOR_RED,
                     'mask-head': COLOR_RED,
                     'face-cap': COLOR_RED,
                     'mask-cap': COLOR_GREED,
                     'uniform': COLOR_GREED,
                     'non-uniform': COLOR_RED}

        for i in range(num_instances):
            cheflable = labels[i].split(" ")[0]
            color = np.array(color_dic[cheflable])
            # print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # print(color, cheflable)
            # ERROR [06/12 11:46:28 daemon    chefCap]: ++++++++++++++++++++++++++++++++++++++++++++++++++++
            # ERROR [06/12 11:46:28 daemon    chefCap]: [0.749 0.749 0.   ]
            # ERROR [06/12 11:46:28 daemon    chefCap]: mask-cap
            # logger.critical("++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # logger.critical(color)
            # logger.critical(labels[i].split(" ")[0])
            #             print(labels[i])
            #             mask-cap 100%
            #             mask-cap 82%
            #             mask-cap 76%
            #             mask-cap 98%
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color)

            # if masks is not None:
            #     for segment in masks[i].polygons:
            #         self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                # elif masks is not None:
                #     x0, y0, x1, y1 = masks[i].bbox()
                #
                #     # draw text in the center (defined by median) when box is not drawn
                #     # median is less sensitive to outliers.
                #     text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                #     horiz_align = "center"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                        instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                        or y1 - y0 < 40 * self.output.scale
                ):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                # lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = (
                        np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                        * 0.5
                        * self._default_font_size
                )
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )

        # draw keypoints
        # if keypoints is not None:
        #     for keypoints_per_instance in keypoints:
        #         self.draw_and_connect_keypoints(keypoints_per_instance)

        return self.output

    def draw_text(
            self,
            text,
            position,
            *,
            font_size=None,
            color="g",
            horizontal_alignment="center",
            rotation=0
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        # color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        # color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output


def base64toImageArray(img_base64):
    img_data = base64.b64decode(img_base64)
    image_np = np.frombuffer(img_data, np.uint8)
    image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    return image_np


def numpArray2Base64(img_arr):
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    img_str = cv2.imencode('.jpg', img_arr)[1].tostring()
    img_base64 = base64.b64encode(img_str).decode('utf-8')
    return img_base64


# x, y, w, h = detection[2][0], \
#                      detection[2][1], \
#                      detection[2][2], \
#                      detection[2][3]
#         xmin, ymin, xmax, ymax = convertBack(
#             float(x), float(y), float(w), float(h))
#
def convertBack(detection):
    x, y, w, h = detection
    x, y, w, h = float(x), float(y), float(w), float(h)
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


class YOLO_single_img():
    def __init__(self, configPath="cfg/yolo-obj.cfg", weightPath="weights/yolo-obj_final.weights",
                 metaPath="cfg/obj.data",
                 gpu_id=1):

        self.metaMain, self.netMain, self.altNames, self.dark = None, None, None, darknet
        # self.logger = setup_logger(log_level=logging.CRITICAL)
        if HAS_GPU:
            set_gpu(gpu_id)
        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(configPath) + "`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(weightPath) + "`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(metaPath) + "`")
        if self.netMain is None:
            self.netMain = darknet.load_net_custom(configPath.encode(
                "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if self.metaMain is None:
            self.metaMain = darknet.load_meta(metaPath.encode("ascii"))
        if self.altNames is None:
            try:
                with open(metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                      re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass
        self.size = (self.dark.network_width(self.netMain),
                     self.dark.network_height(self.netMain))
        self._seconds = 0

    def darkdetect(self, image_src):
        darknet_image = self.dark.make_image(self.dark.network_width(self.netMain),
                                             self.dark.network_height(self.netMain), 3)

        try:
            frame_rgb = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, self.size, interpolation=cv2.INTER_LINEAR)
            # self.logger.info(frame_resized.shape)
            self.dark.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
            detections = self.dark.detect_image(self.netMain, self.metaMain, darknet_image, thresh=0.25)
            return detections, cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
        except:
            raise

    def getsize(self):
        return self.size


# logger = setup_logger(log_level=logging.CRITICAL)
# yoyo = YOLO_single_img(configPath="cfg/chefCap.cfg", weightPath="cfg/chefCap_3000.weights", metaPath="cfg/chefCap.data")
thirteentimestamp = lambda: int(round(time.time() * 1e3))


def kill_duplicate_by_score(prediction, xou_thres=.7):
    from itertools import combinations
    def bb_intersection_over_union(boxA, boxB):
        boxA = (boxA[0], boxA[1], boxA[0] + boxA[2], boxA[3] + boxA[1])
        boxB = (boxB[0], boxB[1], boxB[0] + boxB[2], boxB[3] + boxB[1])
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    prediction[:] = [x for x in prediction if float(x[1]) > .5]

    boxcombins = combinations(prediction, 2)
    for boxcomb in boxcombins:
        try:
            xou = bb_intersection_over_union(boxcomb[0][2], boxcomb[1][2])
            if xou > float(xou_thres):
                prediction.remove(boxcomb[1] if boxcomb[0][1] > boxcomb[1][1] else boxcomb[0])
        except:
            continue

    return prediction
