from ctypes import (
    CDLL,
    Structure,
    RTLD_GLOBAL,
    c_void_p, POINTER, c_char_p, pointer,
    c_int, c_float,
)
import numpy as np


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int)]


class ddx(Structure):
    _fields_ = [("nboxes", c_int),
                ("dd", POINTER(DETECTION))]


thing_classes = ['face-head', 'mask-head', 'face-cap', 'mask-cap', 'uniform', 'non-uniform']
lib = CDLL('libuse.so', RTLD_GLOBAL)
unitest = lib.unitest
unitest.argtypes = [c_char_p, c_char_p, c_char_p, c_int]
unitest.restype = ddx

detect_narray = lib.detect_by_narray
detect_narray.argtypes = [c_char_p, c_char_p, c_char_p, c_int]
detect_narray.restype = ddx


def detect(cfg_name="cfg/chefCap-sam-mish.cfg", weights_name="backup/chefCap-sam-mish_last.weights",
           img_name="test.jpg", gpu_id=0):
    c_name = "../cfg/chefCap-sam-mish.cfg"  # cfg_name
    w_name = "../backup/chefCap-sam-mish_last.weights"  # weights_name
    i_name = "41555ee52bab40169b10253ab30259fa_nondetected.jpg"  # img_name
    ddx = unitest(c_name.encode("ascii"), w_name.encode("ascii"), i_name.encode("ascII"), 0)
    dets = ddx.dd
    res = []
    for j in range(ddx.nboxes):
        for i in range(6):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                nameTag = thing_classes[i]
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    return sorted(res, key=lambda x: -x[1])


def detect_by_narray(cfg_name="cfg/chefCap-sam-mish.cfg", weights_name="backup/chefCap-sam-mish_last.weights",
                     img=np.zeros([30, 30, 3], dtype=np.int), gpu_id=0):
    c_name = "../cfg/chefCap-sam-mish.cfg"  # cfg_name
    w_name = "../backup/chefCap-sam-mish_last.weights"  # weights_name
    i_name = img
    ddx = detect_narray(c_name.encode("ascii"), w_name.encode("ascii"), i_name.tobytes(), 0)
    dets = ddx.dd
    res = []
    for j in range(ddx.nboxes):
        for i in range(6):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                nameTag = thing_classes[i]
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    return sorted(res, key=lambda x: -x[1])


import cv2

size = (576, 576)
image_src = cv2.imread("41555ee52bab40169b10253ab30259fa_nondetected.jpg")
image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
# image_src = cv2.resize(image_src, size, interpolation=cv2.INTER_LINEAR)
print(detect_by_narray(img=image_src, gpu_id=1))

