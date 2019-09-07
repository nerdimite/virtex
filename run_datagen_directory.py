import argparse
import logging
import time

import os

import cv2
import numpy as np
import pandas as pd

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from utils import grid_coords, grid_pixel_search, draw_grid

logger = logging.getLogger('TfPoseEstimator-WebCam')
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
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--folder', type=str, default='./images/', help='folder containing the input images to be traine upon')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')

    parser.add_argument('--tensorrt', type=str, default="False", help='for tensorrt process.')
    parser.add_argument('--pose', type=str, default="Some Pose", help='the name of the pose for which training data is being generated')
    parser.add_argument('--nb-data', type=int, default=10, help='number of training entries to record for training each pose')
    parser.add_argument('--save-to', type=str, default='./data/', help='the directory where the data will be saved')
    parser.add_argument('--save-as', type=str, default='null', help='save the csv data file to')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))

    pose_dataset = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
    def add_data(df, data):
        df = df.append(data, ignore_index=True)
        df = df.replace(np.nan, -1)
        return df


    for in_img in os.listdir(args.folder):
        input_path = os.path.join(args.folder,in_img)
        try:
            image = cv2.imread(input_path)
            image = cv2.resize(image, (640,480))
        except:
            break

        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        logger.debug('postprocess+')
        image, centers = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        # Cam res 640x480
        res = 100
        gx, gy, gpw, gph = grid_coords(res,image)
        #draw_grid(image,gx,gy)
        gpix_locs = {k: grid_pixel_search(v[0],v[1],2,gpw,gph) for k,v in centers.items()}

        print(gpix_locs)
        pose_dataset = add_data(pose_dataset, gpix_locs)

        logger.debug('show+')
        cv2.imshow('tf-pose-estimation result', image)
        cv2.waitKey(3)
        logger.debug('finished+')

    data_path = args.save_to
    filename = args.save_as

    if os.path.isdir(data_path):
        pass
    else:
        os.makedirs(data_path)


    if filename == 'null':
        fnum = 0
        while True:
            fn = '{}-{}.csv'.format(args.pose,fnum)
            dir = os.path.join(data_path,fn)
            if os.path.isfile(dir):
                print('File exists, moving along pose_train-{}.csv'.format(fnum))
                fnum += 1
            else:
                print('Creating new file {}-{}.csv'.format(args.pose,fnum))
                filename = fn
                break

    file_dir = os.path.join(data_path,filename)
    pose_dataset.insert(18,'Pose',args.pose)
    pose_dataset.to_csv(file_dir, index=False)
    cv2.destroyAllWindows()
