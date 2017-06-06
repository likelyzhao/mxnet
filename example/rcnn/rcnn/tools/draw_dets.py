from __future__ import print_function
import argparse
import pprint
import mxnet as mx

from ..config import config, default, generate_config
from ..symbol import *
from ..core.tester import draw_all_detection
import numpy as np
import cPickle
import os
from rcnn.io.rcnn import get_rcnn_testbatch


def _get_max_overlap(gt_roi,det_boxes):

    bbgt = gt_roi
    bb = det_boxes
    if len(det_boxes)>0:
        # compute overlaps
        # intersection
        ixmin = np.maximum(bbgt[0], bb[:,0])
        iymin = np.maximum(bbgt[1], bb[:,1])
        ixmax = np.minimum(bbgt[2], bb[:,2])
        iymax = np.minimum(bbgt[3], bb[:,3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (bbgt[:, 2] - bbgt[:, 0] + 1.) *
               (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
    return ovmax,jmax


def _draw_gtbox(im, gt_box, label,scale):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import cv2
    color_white = (255, 255, 255)
    bbox = gt_box[:4] * scale
    score = gt_box[-1]
    bbox = map(int, bbox)
    cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color_white, thickness=2)
    cv2.putText(im, '%s %.3f' % (label, score), (bbox[0], bbox[1] + 10),
                color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im


def draw_dets(dataset, image_set, root_path, dataset_path, dets_path,thresh):
    import cv2
    # print config
    pprint.pprint(config)

    # load symbol and testing data
    # if has_rpn:
    imdb = eval(dataset)(image_set, root_path, dataset_path)
    roidb = imdb.gt_roidb()

    ### load the detres
    assert os.path.exists(dets_path), 'rpn data not found at {}'.format(dets_path)
    with open(dets_path, 'rb') as f:
        all_boxes = cPickle.load(f)


    for i in range(0,len(roidb)):
        roidb = [roidb[i]]
        data, label, im_info = get_rcnn_testbatch(roidb)
        scale = im_info[0, 2]
        gt_rois = data['rois']
        boxes_this_image=[[]]
        for j in range(1,imdb.num_classes):
            if len(all_boxes[j][i])==0:
                continue
            boxes_this_image.appen(all_boxes[j][i].append(j))

        boxes_draw = np.array([all_boxes[k][i] for k in range(1, imdb.num_classes)])
        im = draw_all_detection(data,boxes_draw,imdb.num_classes,scale)

       # boxes_this_image = np.array([all_boxes[j][i] for j in range(1, imdb.num_classes)])
        for j in len(gt_rois):
            overlap, idx = _get_max_overlap(gt_rois[j],boxes_this_image)
            if overlap > thresh :
                if idx == roidb[i]['gt_classes'][j]:
                    cls_right =1
                else:
                    det_right = 1
            else:
                det_miss =1
            im_bk =im.copy()
            _draw_gtbox(im_bk,gt_rois[j],roidb[i]['gt_classes'][j],scale)
            image_name = roidb[i]['image'] + "_idx_"+ j + "_label_" + roidb[i]['gt_classes'][j] +".jpg"
            if cls_right:
                savefile = "res/cls_right/" + image_name
            if det_right:
                savefile = "res/det_right/" + image_name
            if det_right:
                savefile = "res/det_miss/" + image_name
            cv2.imsave(im,savefile)




def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    # general
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)
    parser.add_argument('--image_set', help='image_set name', default=default.test_image_set, type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    parser.add_argument('--dets_path', help='dets path', default="", type=str)

    parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    draw_dets(args.dataset, args.image_set, args.root_path, args.dataset_path,args.dets_path,
             args.thresh)

if __name__ == '__main__':
    main()
