import base64
import json
import logging
import os
import sqlite3
import time
import traceback
from io import BytesIO

import numpy as np
from detectron2.structures import Instances
from flask import Flask, request, Response
from termcolor import colored
# import tempfile
# import uuid
# from io import BytesIO
#
from utils import (
    img_to_array_raw,
    load_img,
    myx_Visualizer,
    numpArray2Base64,
    convertBack,
    setup_logger,
    # YOLO_single_img
)

# from tensorflow.keras.preprocessing import image
#
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
YOLO_SIZE = (608, 608)
# yoyo = YOLO_single_img(configPath="cfg/chefCap.cfg", weightPath="cfg/chefCap_3000.weights", metaPath="cfg/chefCap.data")
logger = setup_logger(log_level=logging.DEBUG)
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


class sqliteDealWithvideoStream(object):
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

    def execute(self, sql: str, tablename: str, tupleobject: list = []):
        if self.cursor:
            if sql == 'replace':
                self.cursor.executemany(f'REPLACE INTO {tablename} VALUES (?,?,?,?,?,?)', tupleobject)
                self.conn.commit()
            elif sql == 'select':
                query_result = self.cursor.execute(f'select * from {tablename}')
                return query_result.fetchall()


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


# @app.route('/pictureResults', methods=['POST'])
# def pictureResults():
#     error_type = {'face-head': "0",
#                   'mask-head': "0",
#                   'face-cap': "0",
#                   'mask-cap': "1",
#                   'uniform': "2",
#                   'non-uniform': "3",
#                   }
#     non_mask = ['face-head', 'face-cap']
#     non_cap = ['face-head', 'mask-head']
#     non_uniform = ['non-uniform']
#     thing_classes = ['face-head', 'mask-head', 'face-cap', 'mask-cap', 'uniform', 'non-uniform']
#     try:
#         deviceSn = request.json.get("deviceSn")  # 设备SN号
#         imgbase64 = request.json.get("IMG_BASE64")  # 原始图片BASE64位编码
#         analysistype = request.json.get("ANALYSIS_TYPE")  # 分析算法类型：格式：1|2|3 1.帽子 2.口罩 3.工装
#
#         kitchen_img = img_to_array_raw(load_img(BytesIO(base64.b64decode(imgbase64))), dtype=np.uint8)  # / 255.
#
#         predicts, kitchen_img_resized = yoyo.darkdetect(kitchen_img)
#         non_mask_list, non_cap_list, non_uniform_list = [], [], []
#         typea, typeb, typec = {
#                                   "analysisType": "1",
#                                   "analysisResult": non_cap_list
#                               }, \
#                               {
#                                   "analysisType": "2",
#                                   "analysisResult": non_mask_list
#                               }, \
#                               {
#                                   "analysisType": "3",
#                                   "analysisResult": non_uniform_list
#                               }
#         results = [
#             typea, typeb, typec
#         ]
#         vlz = myx_Visualizer(kitchen_img_resized, {"thing_classes": thing_classes}, instance_mode=1)
#         # "pred_boxes":,"scores","pred_classes"
#         instance = Instances(YOLO_SIZE,
#                              **{"pred_boxes": np.array(list(map(convertBack, [x[2] for x in predicts]))),
#                                 "scores": np.array([x[1] for x in predicts]),
#                                 "pred_classes": np.array([thing_classes.index(x[0]) for x in predicts])})
#         vout = vlz.draw_instance_predictions(predictions=instance)
#         back_img = vout.get_image()
#         logger.debug(instance)
#         # cv2.imwrite('test.jpg', back_img)
#         for i in predicts:
#             l, c, b = i
#             if l in non_cap:
#                 non_cap_list.append({
#                     "score": f"{float(c)}",
#                     "ymin": f"{float(b[1])}",
#                     "xmin": f"{float(b[0])}",
#                     "ymax": f"{float(b[3])}",
#                     "flag": f"{error_type[l]}",
#                     "xmax": f"{float(b[2])}",
#                 })
#             elif l in non_mask:
#                 non_mask_list.append({
#                     "score": f"{float(c)}",
#                     "ymin": f"{float(b[1])}",
#                     "xmin": f"{float(b[0])}",
#                     "ymax": f"{float(b[3])}",
#                     "flag": f"{error_type[l]}",
#                     "xmax": f"{float(b[2])}",
#                 })
#             elif l in non_uniform:
#                 non_uniform_list.append({
#                     "score": f"{float(c)}",
#                     "ymin": f"{float(b[1])}",
#                     "xmin": f"{float(b[0])}",
#                     "ymax": f"{float(b[3])}",
#                     "flag": f"{error_type[l]}",
#                     "xmax": f"{float(b[2])}",
#                 })
#             # result = {}
#             # results.append(result)
#         ret_img = numpArray2Base64(back_img)
#         return Response(json.dumps({
#             "code": " A00000",
#             "msg": "成功",
#             "IMG_BASE64": ret_img,
#             "timeStamp": f"{thirteentimestamp()}",
#             "deviceSn": f"{deviceSn}",
#             "results": results,
#         }, ensure_ascii=False), mimetype='application/json')
#     except Exception as exec:
#         logger.error(traceback.format_exc())
#         logger.error(exec)
#         return Response(json.dumps({
#             "code": " A00001",
#             "msg": "失败",
#         }, ensure_ascii=False), mimetype='application/json')


@app.route('/videoStream', methods=['POST'])
def syncVidelStream():
    try:
        deviceInfos = request.json.get("deviceInfo")
        logger.debug(deviceInfos)
        sqlValues = []
        for deviceInfo in deviceInfos:
            deviceSn, rtmpUrl, analysisType = deviceInfo['deviceSn'], deviceInfo['rtmpUrl'], deviceInfo['analysisType']
            frameTime, startTime, endTime = deviceInfo['frameTime'], deviceInfo['startTime'], deviceInfo['endTime']
            sqlValues.append((deviceSn, rtmpUrl, analysisType, frameTime, startTime, endTime))
        sqliteDealWithvideoStream().execute('replace', 'syncVidelStream', sqlValues)
        return Response(json.dumps({
            "code": "A00000",
            "msg": "Succeed",
        }, ensure_ascii=False), mimetype='application/json')
    except Exception as exec:
        logger.error(traceback.format_exc())
        logger.error(exec)
        return Response(json.dumps({
            "code": "A00001",
            "msg": "Faild",
        }, ensure_ascii=False), mimetype='application/json')


@app.route('/querySyncVidelStream', methods=['GET'])
def index():
    return Response(json.dumps({
        "data": sqliteDealWithvideoStream().execute('select', 'syncVidelStream')
    }, ensure_ascii=False), mimetype='application/json')


@app.route('/test', methods=['POST'])
def testhelloworld():
    logger.info(request.json.get("data"))
    return Response(json.dumps({
        "data": request.json.get("data")
    }, ensure_ascii=False), mimetype='application/json')


def unitest():
    sqlValues = []
    # mac243324342	rtmp://58.254.140.125/live?mkSecret=d647a7909071ba1c40833620eafb1e54&mkTime=1569286862/shenzhenipc003	1|2	600	12:00:00	23:59:59
    for i in range(99):
        sqlValues.append((f'mac2433243{i}',
                          'rtmp://58.254.140.125/live?mkSecret=d647a7909071ba1c40833620eafb1e54&mkTime=1569286862/shenzhenipc003',
                          '1|2',
                          '600', '12:00:00', '23:59:59'))
    sqliteDealWithvideoStream().execute('replace', 'syncVidelStream', sqlValues)


# yoyo = YOLO_single_img()

# yoyo.darkdetect(cv2.imread('mq291bb92e000113-155449.jpg'))
app.run(debug=True, port=5123, host='0.0.0.0')
# if __name__ == '__main__':
#     unitest()
