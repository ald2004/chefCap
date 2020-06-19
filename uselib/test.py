from ctypes import (
    CDLL,
    Structure,
    RTLD_GLOBAL,
    c_void_p, POINTER, c_char_p, pointer,
    c_int, c_float,
)
import cv2


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


lib = CDLL('libuse.so', RTLD_GLOBAL)
unitest = lib.unitest
unitest.argtypes = [c_char_p, c_char_p, c_int, c_int, c_char_p, c_char_p,c_int]
unitest.restype = c_void_p

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

im = cv2.imread("41555ee52bab40169b10253ab30259fa_nondetected.jpg")
frame_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# frame_resized = cv2.resize(frame_rgb, (576, 576), interpolation=cv2.INTER_LINEAR)
unitest("../cfg/chefCap-sam-mish.cfg".encode("ascii"), "../backup/chefCap-sam-mish_last.weights".encode("ascii"), 0, 1,
        "../cfg/chefCap.data".encode("ascII"), "41555ee52bab40169b10253ab30259fa_nondetected.jpg".encode("ascII"),0)
# metaMain = load_meta("../cfg/chefCap.names".encode("ascii"))
# print(metaMain._fields_[0][1])
# [('classes', <class 'ctypes.c_int'>), ('names', <class '__main__.LP_c_char_p'>)]
