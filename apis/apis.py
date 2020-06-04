import sqlite3
import base64
import json
import os
import logging
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
)
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
            # 1	'口罩分析'	'0'
            # 2	'帽子分析'	'1'
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
            return detections
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
    try:
        deviceSn = request.json.get("deviceSn")  # 设备SN号
        imgbase64 = request.json.get("IMG_BASE64")  # 原始图片BASE64位编码
        analysistype = request.json.get("ANALYSIS_TYPE")  # 分析算法类型：格式：1|2|3 1.帽子 2.口罩 3.工装

        kitchen_img = img_to_array_raw(load_img(BytesIO(base64.b64decode(imgbase64))), dtype=np.uint8)  # / 255.
        return Response(json.dumps({
            "code": " A00000",
            "msg": "成功",
            "results": yoyo.darkdetect(kitchen_img)
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
# yoyo.darkdetect(cv2.imread('mq291bb92e000113-155449.jpg'))
app.run(debug=True, port=5123, host='0.0.0.0')
