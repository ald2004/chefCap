import sqlite3
import base64
import json
import os
import logging
import time
import traceback
import darknet
from darknet import set_gpu
import sys
from fvcore.common.timer import Timer
from termcolor import colored
# import tempfile
# import uuid
# from io import BytesIO
#
import cv2
from utils import (
    setup_logger,
    img_to_array_raw,
    load_img,
    myx_Visualizer,
    numpArray2Base64,
    convertBack,
)
from detectron2.structures import Instances
import numpy as np
from io import BytesIO
# # tf
# import tensorflow as tf
# from detectron2.config import get_cfg
# from detectron2.data import (
#     DatasetCatalog,
#     MetadataCatalog,
# )
# from detectron2.data.detection_utils import read_image
# # pytorch
# from detectron2.engine import DefaultPredictor
# from detectron2.utils.visualizer import ColorMode, Visualizer
from flask import Flask, request, Response

# from tensorflow.keras.preprocessing import image
#
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
YOLO_SIZE = (608, 608)
# output_saved_model_dir = './tensorrt_dir'
# # from flask_cors import CORS
# img_size = 256
# _SMALL_OBJECT_AREA_THRESH = 1000
app = Flask(__name__)


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


class sqliteDealWith(object):
    '''
        CREATE TABLE "main"."analysisType" (
    "typeId"  INTEGER NOT NULL,
    "typeName"  TEXT NOT NULL,
    "modifyTag"  TEXT NOT NULL,
    PRIMARY KEY ("typeId" ASC));

    '''

    def __init__(self):
        self.conn = sqlite3.connect('chefCap.db')
        self.cursor = self.conn.cursor()

    def __enter__(self):
        if not self.conn:
            self.conn = sqlite3.connect('chefCap.db')
            self.cursor = self.conn.cursor()
        return self

    def __exit__(self, errorType, errorValue, errorTrace):
        self.cursor.close()
        self.conn.close()

    def execute(self, sql: str, tablename: str, tupleobject: list):
        if self.cursor:
            # 1 '口罩分析'      '0'
            # 2 '帽子分析'      '1'
            # tupleobject =   [(96, 'xxxxxxxxx', '96'),
            #                  (97, 'xxxxxxxxx', '97'),
            #                  (98, 'xxxxxxxxx', '98')]
            # c.executemany('INSERT INTO stocks VALUES (?,?,?,?,?)', purchases)
            if sql == 'replace':
                self.cursor.executemany(f'REPLACE INTO {tablename} VALUES (?,?,?)', tupleobject)
                self.conn.commit()


class YOLO_single_img():
    def __init__(self, configPath="cfg/yolo-obj.cfg", weightPath="weights/yolo-obj_final.weights",
                 metaPath="cfg/obj.data",
                 gpu_id=1):

        self.metaMain, self.netMain, self.altNames, self.dark, self.tt = None, None, None, darknet, Timer()
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

    def darkdetect(self, image_src):
        darknet_image = self.dark.make_image(self.dark.network_width(self.netMain),
                                             self.dark.network_height(self.netMain), 3)
        self.tt.reset()
        try:
            # frame_rgb = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(image_src,
                                       (self.dark.network_width(self.netMain),
                                        self.dark.network_height(self.netMain)),
                                       interpolation=cv2.INTER_LINEAR)
            logger.info(frame_resized.shape)
            self.dark.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
            detections = self.dark.detect_image(self.netMain, self.metaMain, darknet_image, thresh=0.25)
            logger.info(detections)
            return detections, frame_resized
        except:
            raise


@app.route('/getAnalysisType', methods=['POST'])
def getAnalysisType():
    try:
        # print(request.json)
        # print('-------------')
        requestdata = request.json.get("analysisType")
        analysisType = []
        analysisType.extend((x['typeId'], x['typeName'], x['modifyTag']) for x in requestdata)
        sqliteDealWith().execute('replace', 'analysisType', analysisType)
        return Response(json.dumps({
            "code": " A00000",
            "msg": "成功",
        }, ensure_ascii=False), mimetype='application/json')
    except Exception as exc:
        logger.error(traceback.format_exc())
        logger.error(exec)
        return Response(json.dumps({
            "code": " A00001",
            "msg": "失败",
        }, ensure_ascii=False), mimetype='application/json')


@app.route('/pictureResults', methods=['POST'])
def pictureResults():
    error_type = {'face-head': "0",
                  'mask-head': "0",
                  'face-cap': "0",
                  'mask-cap': "1",
                  'uniform': "2",
                  'non-uniform': "3",
                  }
    non_mask = ['face-head', 'face-cap']
    non_cap = ['face-head', 'mask-head']
    non_uniform = ['non-uniform']
    thing_classes = ['face-head', 'mask-head', 'face-cap', 'mask-cap', 'uniform', 'non-uniform']
    try:
        deviceSn = request.json.get("deviceSn")  # 设备SN号
        imgbase64 = request.json.get("IMG_BASE64")  # 原始图片BASE64位编码
        analysistype = request.json.get("ANALYSIS_TYPE")  # 分析算法类型：格式：1|2|3 1.帽子 2.口罩 3.工装

        kitchen_img = img_to_array_raw(load_img(BytesIO(base64.b64decode(imgbase64))), dtype=np.uint8)  # / 255.

        predicts, kitchen_img_resized = yoyo.darkdetect(kitchen_img)
        non_mask_list, non_cap_list, non_uniform_list = [], [], []
        typea, typeb, typec = {
                                  "analysisType": "1",
                                  "analysisResult": non_cap_list
                              }, \
                              {
                                  "analysisType": "2",
                                  "analysisResult": non_mask_list
                              }, \
                              {
                                  "analysisType": "3",
                                  "analysisResult": non_uniform_list
                              }
        results = [
            typea, typeb, typec
        ]
        vlz = myx_Visualizer(kitchen_img_resized, {"thing_classes": thing_classes}, instance_mode=1)
        # "pred_boxes":,"scores","pred_classes"
        instance = Instances(YOLO_SIZE,
                             **{"pred_boxes": np.array(list(map(convertBack, [x[2] for x in predicts]))),
                                "scores": np.array([x[1] for x in predicts]),
                                "pred_classes": np.array([thing_classes.index(x[0]) for x in predicts])})
        vout = vlz.draw_instance_predictions(predictions=instance)
        back_img = vout.get_image()
        logger.debug(instance)
        # cv2.imwrite('test.jpg', back_img)
        for i in predicts:
            l, c, b = i
            if l in non_cap:
                non_cap_list.append({
                    "score": f"{float(c)}",
                    "ymin": f"{float(b[1])}",
                    "xmin": f"{float(b[0])}",
                    "ymax": f"{float(b[3])}",
                    "flag": f"{error_type[l]}",
                    "xmax": f"{float(b[2])}",
                })
            elif l in non_mask:
                non_mask_list.append({
                    "score": f"{float(c)}",
                    "ymin": f"{float(b[1])}",
                    "xmin": f"{float(b[0])}",
                    "ymax": f"{float(b[3])}",
                    "flag": f"{error_type[l]}",
                    "xmax": f"{float(b[2])}",
                })
            elif l in non_uniform:
                non_uniform_list.append({
                    "score": f"{float(c)}",
                    "ymin": f"{float(b[1])}",
                    "xmin": f"{float(b[0])}",
                    "ymax": f"{float(b[3])}",
                    "flag": f"{error_type[l]}",
                    "xmax": f"{float(b[2])}",
                })
            # result = {}
            # results.append(result)
        ret_img = numpArray2Base64(back_img)
        return Response(json.dumps({
            "code": " A00000",
            "msg": "成功",
            "IMG_BASE64": ret_img,
            "timeStamp": f"{thirteentimestamp()}",
            "deviceSn": f"{deviceSn}",
            "results": results,
        }, ensure_ascii=False), mimetype='application/json')
    except Exception as exec:
        logger.error(traceback.format_exc())
        logger.error(exec)
        return Response(json.dumps({
            "code": " A00001",
            "msg": "失败",
        }, ensure_ascii=False), mimetype='application/json')


def unitest():
    xxx = []
    for i in range(99):
        xxx.append((i, 'xxxxxxxxx', f'{i}'))
    a = sqliteDealWith()
    a.execute('replace', 'analysisType', xxx)


logger = setup_logger()
yoyo = YOLO_single_img()
thirteentimestamp = lambda: int(round(time.time() * 1e3))
# yoyo.darkdetect(cv2.imread('mq291bb92e000113-155449.jpg'))
app.run(debug=True, port=5123, host='0.0.0.0')
