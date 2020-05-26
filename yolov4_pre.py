import xml.etree.ElementTree as ET
import os
import json
import glob
from fvcore.common.file_io import PathManager


def convert_xminymin_xcenterycenter(h, w, xmin, ymin, xmax, ymax):
    # < x_center > < y_center > < width > < height > - float values relative to width and height of image, it can  be  equal from (0.0 to 1.0]
    dw = 1. / (float(w))
    dh = 1. / (float(h))
    x = (xmin + xmax) / 2.0

    y = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    #     return x, y, w, h
    return f'{x} {y} {w} {h}'


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


things_class_dict = {
    'face-head': 0,
    'mask-head': 1,
    'face-cap': 2,
    'mask-cap': 3
}

  
def unitest():
    '''
    <xmin>206.42315644383186</xmin>\n',
    <ymin>427.1415441176471</ymin>\n',
    <xmax>288.4631288766368</xmax>\n',
    <ymax>514.4944852941177</ymax>\n',
    h 720
    w 1280
    '''
    x, y, w, h = convert_xminymin_xcenterycenter(720, 1280, 206.423, 427.141, 288.463, 514.494)
    print(x, y, w, h)
    xmin, ymin, xmax, ymax = convertBack(x, y, w, h)
    print(xmin, ymin, xmax, ymax)

# unitest()
xmlfiles = glob.glob("data/Annotations/*.xml")
for xmlfile in xmlfiles:
    with PathManager.open(xmlfile) as fid:
        tree = ET.parse(fid)
        # < name > mask - cap < / name >
        # < filename > mqxxxxxxxx - 065435.jpg < / filename >
        filename_ = tree.find("filename").text.strip()
        width_ = tree.find("size").find("width").text.strip()
        height_ = tree.find("size").find("height").text.strip()
        depth_ = tree.find("size").find("depth").text.strip()
        txtfile=filename_.split('.')[0]+'.txt'
        with PathManager.open(os.path.join('data','txt',txtfile),mode='w') as fid:
            for obj in tree.findall("object"):
                oclass_ = things_class_dict[obj.find("name").text.strip()]
                bbox = obj.find("bndbox")
                try:
                    bbox_: str = convert_xminymin_xcenterycenter(height_,width_,
                        float(bbox.find("xmin").text),
                        float(bbox.find("ymin").text),
                        float(bbox.find("xmax").text),
                        float(bbox.find("ymax").text)
                    )
                except:
                    raise
                fid.write(f'{oclass_} {bbox_}\n')
#                 print(filename_,oclass_,bbox_)
#     txtfile=filename_.split('.')[0]+'.txt'
#     with PathManager.open(os.path.join('data','txt',txtfile),mode='w') as fid:
#         fid.write(f'{oclass_} {bbox_}')
#         break

