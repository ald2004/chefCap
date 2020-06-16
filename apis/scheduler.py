import functools
import json
import logging
import os
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime

import cv2
import numpy as np
import pysftp
import pytz
import requests
from detectron2.structures import Instances
from detectron2.utils import comm
from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager
from utils import (
    setup_logger,
    YOLO_single_img,
    myx_Visualizer,
    convertBack,
    thirteentimestamp,
    kill_duplicate_by_score
)

logger = setup_logger(log_level=logging.DEBUG)
yoyo = YOLO_single_img(configPath="cfg/chefCap.cfg", weightPath="cfg/chefCap_diounms_mosaic_20000.weights",
                       metaPath="cfg/chefCap.data")
# yoyo = YOLO_single_img(configPath="cfg/chefCap.cfg", weightPath="cfg/chefCap_21000.weights",metaPath="cfg/chefCap.data")
# yoyo = YOLO_single_img(configPath="cfg/chefCap.cfg", weightPath="cfg/chefCap_12000.weights", metaPath="cfg/chefCap.data")
# yoyo = YOLO_single_img(configPath="cfg/chefCap.cfg", weightPath="cfg/chefCap_final.weights", metaPath="cfg/chefCap.data")
API_ENDPOINT = "http://10.1.198.6:5123/querySyncVidelStream"
API_ENDPOINT_SEND = "http://10.1.198.6:9906/asiainfoAI/streamingResults"
thing_classes = ['face-head', 'mask-head', 'face-cap', 'mask-cap', 'uniform', 'non-uniform']
# WHERE_TO_UPLOAD_TO = '/home/puaiuc/images/analysisImgs'
WHERE_TO_UPLOAD_TO = '/home/nginx/images/analysisImgs'
last_check_table = {"mq289vcee5000015": 600}


def add_time():
    while True:
        for k, v in last_check_table.items():
            last_check_table[k] = v + 1
            # print(f'{k}----{last_check_table[k]}')
        time.sleep(1)


def read_time():
    while True:
        # print(last_check_table)
        time.sleep(100)


def upload(back_img):
    try:
        srv = pysftp.Connection(host="10.1.198.6", username="yuanpu",
                                password="111111", log="./logs/pysftp.log")
        with tempfile.TemporaryDirectory() as tempdirname:
            tempfilename = f'{uuid.uuid4().hex}.jpg'
            tempfullname = os.path.join(tempdirname, tempfilename)
            logger.debug(
                f'+++++++++++++++ tempfullname is {tempfullname} back_img.shape {back_img.shape},+++++++++++++++')
            cv2.imwrite(tempfullname, back_img)
            with srv.cd(WHERE_TO_UPLOAD_TO):  # chdir to public
                srv.put(tempfullname)  # upload file to nodejs/
        return tempfilename
    except Exception as exec:
        logger.error(traceback.format_exc())
        logger.error(exec)
        return None


def do_detect_upload(rtmpurl: str, analysisType='1|2|3'):
    try:
        cap = cv2.VideoCapture(rtmpurl)
        # import random
        # cap = cv2.VideoCapture("logs/video/mq17cc36c1000440-20200608182458.mp4")
        # cap.set(1, random.choice(range(int(float(cap.get(cv2.CAP_PROP_FRAME_COUNT))))))  # Out[5]: 108181.0
        # frameRate = int(np.floor(cap.get(cv2.CAP_PROP_FPS)))
        # frameRate = frameRate if frameRate else 20
        # count = 0
        logger.debug(f'start det {rtmpurl} ...')
        # cap.set(3, yoyo.getsize()[0])  # w
        # cap.set(4, yoyo.getsize()[1])  # h
        # while True:
        # prev_time.reset()
        ret, frame_read = cap.read()
        if not ret:
            logger.debug("ret is not...")
            cap.release()
            return None, None

        tt = Timer()
        predicts, kitchen_img_resized = yoyo.darkdetect(frame_read)
        # ('uniform', 0.9847872257232666, (226.92221069335938, 266.7281188964844, 87.1346435546875, 198.78860473632812))
        analysisType_predicts = []
        predicts_killed = kill_duplicate_by_score(predicts)
        for xx in analysisType.split('|'):
            if xx == '1':
                for yy in predicts_killed:
                    if yy[0] in ['face-head', 'face-cap', 'mask-cap']:
                        analysisType_predicts.append(yy)
            elif xx == '2':
                for yy in predicts_killed:
                    if yy[0] in ['face-head', 'mask-cap', 'mask-head']:
                        analysisType_predicts.append(yy)
            elif xx == '3':
                for yy in predicts_killed:
                    if yy[0] in ['non-uniform', 'uniform']:
                        analysisType_predicts.append(yy)
        # predicts = kill_duplicate_by_score(analysisType_predicts)
        predicts = analysisType_predicts
        tt.pause()
        logger.info(f'************** one shot detect time is {tt.seconds()} **************')
        vlz = myx_Visualizer(kitchen_img_resized, {"thing_classes": thing_classes}, instance_mode=1)
        # "pred_boxes":,"scores","pred_classes"
        instance = Instances(yoyo.getsize(),
                             **{"pred_boxes": np.array(list(map(convertBack, [x[2] for x in predicts]))),
                                "scores": np.array([x[1] for x in predicts]),
                                "pred_classes": np.array([thing_classes.index(x[0]) for x in predicts])})
        vout = vlz.draw_instance_predictions(predictions=instance)
        kitchen_img_resized = vout.get_image()

        logger.debug(instance)
        # count += frameRate
        # cap.set(1, count)
        cap.release()
        logger.debug(f'end det {rtmpurl} ...')
        tempfilename = upload(kitchen_img_resized)
        if len(tempfilename):
            cv2.imwrite(f'imgslogs/{tempfilename}', kitchen_img_resized)
            cv2.imwrite(f'imgslogs/nondetected/{tempfilename[:-4]}_nondetected.jpg', frame_read)
            with PathManager.open(f'imgslogs/predictions/{tempfilename[:-4]}.txt', 'w') as fid:
                fid.writelines(f'{predicts}')
        else:
            cv2.imwrite(f'imgslogs/{uuid.uuid4().hex}.jpg', kitchen_img_resized)
            cv2.imwrite(f'imgslogs/{uuid.uuid4().hex}_nondetected.jpg', frame_read)
        return predicts, tempfilename
    except:
        logger.error(traceback.format_exc())
        logger.error(exec)
        logger.debug(f'end det {rtmpurl} with errors...')
        cap.release()
        return None, None


def grab_and_analysis(deviceSn: str, rtmpurl: str, frameTime: str, analysisType='1|2|3') -> dict:
    secs = last_check_table.get(deviceSn)
    logger.debug(f'+++++++++++++++ Starting grab {deviceSn} after {secs} while thres {frameTime}+++++++++++++++')
    if secs:
        if secs >= int(float(frameTime)):
            # if secs >= 60:
            predicts, uploadedurl = do_detect_upload(rtmpurl, analysisType)
            logger.debug(
                f'+++++++++++++++ return from do_detect_upload [{predicts}] and [{uploadedurl}]+++++++++++++++ ')
            if predicts is None or uploadedurl is None:
                return None
            last_check_table[deviceSn] = 0
            if not predicts or not uploadedurl:
                return {}

            if predicts and uploadedurl:
                analysisResult_list_one = []
                analysisResult_list_two = []
                analysisResult_list_three = []
                yy_one = {
                    "analysisType": "1",
                    "imgUrl": f'http://10.1.198.6:18090/analysisImgs/{uploadedurl}',
                    # http://10.1.198.6:18090/analysisImgs/xxx.jpg?
                    "analysisResult": analysisResult_list_one
                }
                yy_two = {
                    "analysisType": "2",
                    "imgUrl": f'http://10.1.198.6:18090/analysisImgs/{uploadedurl}',
                    "analysisResult": analysisResult_list_two
                }
                yy_three = {
                    "analysisType": "3",
                    "imgUrl": f'http://10.1.198.6:18090/analysisImgs/{uploadedurl}',
                    "analysisResult": analysisResult_list_three
                }
                for i in predicts:
                    if i[0] in ['face-head', 'face-cap', 'mask-cap']:
                        analysisResult_list_one.append({
                            "score": f"{i[1]}",
                            "xmin": f"{i[2][0]}",
                            "ymin": f"{i[2][1]}",
                            "xmax": f"{i[2][2]}",
                            "ymax": f"{i[2][3]}",
                            "flag": "0" if i[0] == ['face-head'] else "1"

                        })
                    if i[0] in ['face-head', 'mask-cap', 'mask-head']:
                        analysisResult_list_two.append({
                            "score": f"{i[1]}",
                            "xmin": f"{i[2][0]}",
                            "ymin": f"{i[2][1]}",
                            "xmax": f"{i[2][2]}",
                            "ymax": f"{i[2][3]}",
                            "flag": "0" if i[0] == ['face-head'] else "1"
                        })
                    if i[0] in ['non-uniform', 'uniform']:
                        analysisResult_list_three.append({
                            "score": f"{i[1]}",
                            "xmin": f"{i[2][0]}",
                            "ymin": f"{i[2][1]}",
                            "xmax": f"{i[2][2]}",
                            "ymax": f"{i[2][3]}",
                            "flag": "0" if i[0] == ['non-uniform'] else "1"
                        })
                final_result = []
                for x in analysisType.split('|'):
                    if x == '1':
                        final_result.append(yy_one)
                    elif x == '2':
                        final_result.append(yy_two)
                    elif x == '3':
                        final_result.append(yy_three)
                return {"timeStamp": thirteentimestamp(),
                        "deviceSn": f"{deviceSn}",
                        "result": final_result,
                        }
        else:
            return {}
    else:
        last_check_table[deviceSn] = 600


# @functools.lru_cache(maxsize=10)
def querySyncVidelStream() -> dict:
    r = requests.get(url=API_ENDPOINT)
    return json.loads(r.text)


def d():
    logger.debug('+++++++++++++++ Starting +++++++++++++++')
    while True:
        result_data = querySyncVidelStream()
        logger.debug(f'+++++++++++++++ {result_data} +++++++++++++++')
        for c in result_data['data']:
            deviceSn, rtmpUrl, analysisType, frameTime, startTime, endTime = c
            start_hour, end_hour = datetime.strptime(startTime, "%H:%M:%S").hour, \
                                   datetime.strptime(endTime, "%H:%M:%S").hour
            if not (datetime.now(tz=pytz.timezone('Asia/Shanghai')).hour > start_hour and datetime.now(
                    tz=pytz.timezone('Asia/Shanghai')).hour < end_hour):
                logger.debug('sleepy time seeu...')
                time.sleep(1800)
            else:
                try:
                    result_dict = grab_and_analysis(deviceSn, rtmpUrl, frameTime, analysisType=analysisType)
                    if result_dict:
                        logger.debug(f'+++++++++++++++ {result_dict} +++++++++++++++')
                        r = requests.post(url=API_ENDPOINT_SEND, json=result_dict)
                        ret = json.loads(r.text)
                        logger.info(ret)
                        if ret["code"] != "A00000":
                            logger.error("something wrong...")
                            continue
                    else:
                        logger.error("grab_and_analysis not work continued!!!")
                        continue
                except:
                    logger.error(traceback.format_exc())
                    logger.error(exec)
                    logger.debug(f'post to {API_ENDPOINT_SEND}  with errors...')
                    continue
            time.sleep(5)
            logger.debug('done for now...')
        time.sleep(2)


def unitest():
    td = threading.Thread(name='reader', target=read_time)  # , args=(last_check_table,))
    tt = threading.Thread(name='timer', target=add_time)  # , args=(last_check_table,))
    # tt.setDaemon(True)
    # td.setDaemon(True)
    td.start()
    tt.start()


if __name__ == '__main__':
    td = threading.Thread(name='daemon', target=d)
    tt = threading.Thread(name='timer', target=add_time)
    tr = threading.Thread(name='reader', target=read_time)
    tt.setDaemon(True)
    td.setDaemon(True)
    tr.setDaemon(True)

    td.start()
    tt.start()

    logger.info(comm.get_world_size())
    logger.info(comm.get_local_rank())
    logger.info(comm.get_rank())
    logger.info(threading.active_count())
    logger.info(threading.activeCount())
    # tr.start()
    # print('ss')
    # unitest()
    while True:
        time.sleep(1000)
