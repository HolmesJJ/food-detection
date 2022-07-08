import os
import cv2
import json
import tornado.web
import tornado.ioloop

from base64 import b64decode
from typing import Optional, Awaitable
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

global PORT
global metadata
global predictor

PATH = {
    'train': 'dataset/vegetables/train',
    'coco_train': 'dataset/vegetables/coco/train.json',
    'faster_rcnn_X_101_32x8d_FPN_3x': 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml',
    'model': 'output/model_final.pth'
}


def detect(path):
    im = cv2.imread(path)
    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")
    pred_classes = instances.pred_classes.tolist()
    pred_scores = instances.scores.tolist()
    pred_class_names = [metadata.thing_classes[x] for x in pred_classes]
    pred_boxes = instances.pred_boxes.tensor.numpy().tolist()
    results = []
    for i in range(len(pred_classes)):
        item = {
            'name': pred_class_names[i],
            'score': pred_scores[i],
            'box': pred_boxes[i]
        }
        results.append(item)
    return results


def init():
    register_coco_instances('train_dataset', {}, PATH['coco_train'], PATH['train'])
    mc = MetadataCatalog.get("train_dataset")
    dc = DatasetCatalog.get("train_dataset")  # Must include
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(PATH['faster_rcnn_X_101_32x8d_FPN_3x']))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    cfg.MODEL.WEIGHTS = PATH['model']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    dp = DefaultPredictor(cfg)
    return mc, dp


class BaseRequestHandler(tornado.web.RequestHandler):

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Credentials", "true")
        self.set_header("Access-Control-Allow-Headers", "*")
        self.set_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, PATCH, OPTIONS')

    # vue一般需要访问options方法， 如果报错则很难继续，所以只要通过就行了，当然需要其他逻辑就自己控制。
    # 因为vue访问时，会先执行一次预加载，直接放过就好了
    def options(self):
        # 这里的状态码一定要设置200
        self.set_status(200)
        self.finish()


class DetectRequestHandler(BaseRequestHandler):

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def post(self):
        data = json.loads(self.request.body.decode('utf-8'))
        img = data['image']
        if not os.path.isdir('detect'):
            os.makedirs('detect')
        try:
            file_bytes = open(f'detect/detect.png', 'wb')
            file_bytes.write(b64decode(img))
            file_bytes.close()
            results = detect(f'detect/detect.png')
            response = {
                'code': 0,
                'data': results
            }
        except:
            response = {
                'code': 1,
            }
        self.set_header('Content-type', 'application/json')
        self.write(bytes(json.dumps(response), 'utf-8'))


if __name__ == "__main__":
    metadata, predictor = init()
    app = tornado.web.Application([
        ("/detect", DetectRequestHandler),
    ])
    PORT = 8000
    app.listen(PORT)
    print("Listening on port %s" % PORT)
    tornado.ioloop.IOLoop.instance().start()
