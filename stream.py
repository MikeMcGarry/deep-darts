import argparse
from yacs.config import CfgNode as CN
import os.path as osp
import os
from dataloader import get_splits
import cv2
import numpy as np
from time import time
from dataset.annotate import draw, get_dart_scores
import pickle
from predict import bboxes_to_xy

def predict_stream(yolo):

    cam = cv2.VideoCapture(0)
    i = 0

    while True:
        check, frame = cam.read()
        i += 1

        if i % 60 == 0:
            # Resize frame to 800x800
            img = cv2.resize(frame, (800, 800))
            bboxes = yolo.predict(img)
            preds = bboxes_to_xy(bboxes, 3)
            xy = preds
            xy = xy[xy[:, -1] == 1]
            img = draw(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), xy[:, :2], cfg, circles=False, score=True)
            cv2.imshow('video', img)

            key = cv2.waitKey(1)
            if key == 'z':
                break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    from train import build_model
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='deepdarts_utrecht')
    args = parser.parse_args()

    cfg = CN(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', args.cfg + '.yaml'))
    cfg.model.name = args.cfg

    yolo = build_model(cfg)
    yolo.load_weights(osp.join('models', args.cfg, 'weights'), cfg.model.weights_type)

    predict_stream(yolo)
