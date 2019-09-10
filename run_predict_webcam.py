import argparse
import logging
import time
import os

import cv2
import numpy as np
import pandas as pd

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from utils import grid_coords, grid_pixel_search, draw_grid, load_obj

logger = logging.getLogger('Virtex')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='virtex inference in realtime')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--tensorrt', type=str, default="False", help='for tensorrt process.')
    parser.add_argument('--classifier', type=str, default='./classifiers/classifier.model', help='the path to the classifier model')
    parser.add_argument('--cls-dict', type=str, help='the path to the file containing the mapping from encoded values to class names')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))


    pix_df = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
    def process_stream(df, data):
        X = df.append(data, ignore_index=True)
        X = X.values
        return X

    classifier = load_obj(args.classifier)
    try:
        legend = load_obj(args.cls_dict)
    except:
        pass

    while True:

        ret_val, image = cam.read()

        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        logger.debug('postprocess+')
        image, centers = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        # Pose Prediction
        res = 100
        gx, gy, gpw, gph = grid_coords(res,image)

        gpix_locs = {k: grid_pixel_search(v[0],v[1],2,gpw,gph) for k,v in centers.items()}
        print(gpix_locs)

        x_locs = process_stream(pix_df, gpix_locs)
        y_pred = classifier.predict(x_locs)
        print(y_pred)
        try:
            nm_pose = legend[y_pred[0]]
        except:
            nm_pose = y_pred[0]

        logger.debug('show+')
        cv2.putText(image,
                    "FPS: {} Pose: {}".format((1.0 / (time.time() - fps_time)), nm_pose),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('Virtex Inference', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
