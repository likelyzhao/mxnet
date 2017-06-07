from __future__ import print_function
import argparse
import pprint
import mxnet as mx

from ..config import config, default, generate_config
from ..symbol import *
from ..dataset import *
from ..core.tester import draw_all_detection
import numpy as np
import cPickle
import os
from rcnn.io.rcnn import get_rcnn_testbatch


imagenet_cls_names = np.array(['__background__',\
                         'accordion', 'airplane', 'ant', 'antelope', 'apple', 'armadillo', 'artichoke',\
                         'axe', 'baby_bed', 'backpack', 'bagel', 'balance_beam', 'banana', 'band_aid',\
                         'banjo', 'baseball', 'basketball', 'bathing_cap', 'beaker', 'bear', 'bee',\
                         'bell_pepper', 'bench', 'bicycle', 'binder', 'bird', 'bookshelf', 'bow_tie',\
                         'bow', 'bowl', 'brassiere', 'burrito', 'bus', 'butterfly', 'camel', 'can_opener',\
                         'car', 'cart', 'cattle', 'cello', 'centipede', 'chain_saw', 'chair', 'chime',\
                         'cocktail_shaker', 'coffee_maker', 'computer_keyboard', 'computer_mouse', 'corkscrew',\
                         'cream', 'croquet_ball', 'crutch', 'cucumber', 'cup_or_mug', 'diaper', 'digital_clock',\
                         'dishwasher', 'dog', 'domestic_cat', 'dragonfly', 'drum', 'dumbbell', 'electric_fan',\
                         'elephant', 'face_powder', 'fig', 'filing_cabinet', 'flower_pot', 'flute', 'fox',\
                         'french_horn', 'frog', 'frying_pan', 'giant_panda', 'goldfish', 'golf_ball', 'golfcart',\
                         'guacamole', 'guitar', 'hair_dryer', 'hair_spray', 'hamburger', 'hammer', 'hamster',\
                         'harmonica', 'harp', 'hat_with_a_wide_brim', 'head_cabbage', 'helmet', 'hippopotamus',\
                         'horizontal_bar', 'horse', 'hotdog', 'iPod', 'isopod', 'jellyfish', 'koala_bear', 'ladle',\
                         'ladybug', 'lamp', 'laptop', 'lemon', 'lion', 'lipstick', 'lizard', 'lobster', 'maillot',\
                         'maraca', 'microphone', 'microwave', 'milk_can', 'miniskirt', 'monkey', 'motorcycle',\
                         'mushroom', 'nail', 'neck_brace', 'oboe', 'orange', 'otter', 'pencil_box', 'pencil_sharpener',\
                         'perfume', 'person', 'piano', 'pineapple', 'ping-pong_ball', 'pitcher', 'pizza', 'plastic_bag',\
                         'plate_rack', 'pomegranate', 'popsicle', 'porcupine', 'power_drill', 'pretzel', 'printer', 'puck',\
                         'punching_bag', 'purse', 'rabbit', 'racket', 'ray', 'red_panda', 'refrigerator', 'remote_control',\
                         'rubber_eraser', 'rugby_ball', 'ruler', 'salt_or_pepper_shaker', 'saxophone', 'scorpion',\
                         'screwdriver', 'seal', 'sheep', 'ski', 'skunk', 'snail', 'snake', 'snowmobile', 'snowplow',\
                         'soap_dispenser', 'soccer_ball', 'sofa', 'spatula', 'squirrel', 'starfish', 'stethoscope',\
                         'stove', 'strainer', 'strawberry', 'stretcher', 'sunglasses', 'swimming_trunks', 'swine',\
                         'syringe', 'table', 'tape_player', 'tennis_ball', 'tick', 'tie', 'tiger', 'toaster',\
                         'traffic_light', 'train', 'trombone', 'trumpet', 'turtle', 'tv_or_monitor', 'unicycle', 'vacuum',\
                         'violin', 'volleyball', 'waffle_iron', 'washer', 'water_bottle', 'watercraft', 'whale', 'wine_bottle',\
                         'zebra'])

def _mkdir(path):

    try:
        os.mkdir(path)
    except Exception:
        print(Exception) 

def _get_max_overlap(gt_roi,det_boxes,scale):

    bbgt = gt_roi
    bb = det_boxes* scale
    ovmax =0
    jmax =-1
    if len(det_boxes)>0:
        print("bbshape")
        print(bb.shape)
        print("gt")
        print(bbgt)
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
        uni = ((bb[:,2] - bb[:,0] + 1.) * (bb[:,3] - bb[:,1] + 1.) +
               (bbgt[2] - bbgt[ 0] + 1.) *
               (bbgt[ 3] - bbgt[1] + 1.) - inters)

        overlaps = inters / uni
    #    print(overlaps)
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
    print(gt_box)
    print(label)
    color_white = (255, 255, 255)
    bbox = gt_box[:4] * scale
    score = gt_box[-1]
    bbox = map(int, bbox)
    cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color_white, thickness=2)
    cv2.putText(im, '%s %.3f' % (label, score), (bbox[0], bbox[1] + 10),
                color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im


def draw_dets(dataset, image_set, root_path, dataset_path, dets_path,thresh,save_path):
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
        roidb_now = [roidb[i]]
        data, label, im_info = get_rcnn_testbatch(roidb_now)
        scale = im_info[0, 2]
        gt_rois = data['rois']
        boxes_this_image=[]
        for j in range(1,imdb.num_classes):
            if len(all_boxes[j][i])==0:
                continue
            #boxes_this_image.append(all_boxes[j][i].append(j))
            for box in all_boxes[j][i]:
       #         print(box.tolist())
#                print(np.append[box,[j]])
                boxlist = box.tolist()
                if boxlist[4] <= 0.2:
                   continue
                boxlist.append(j)

#                print(boxlist)

                boxes_this_image.append(boxlist)


        boxes_draw = np.array( [[]] + [all_boxes[k][i] for k in range(1, imdb.num_classes)])
#        print(all_boxes)
#        boxes_this_image = [[]] + [all_boxes[j][i] for j in range(1, imdb.num_classes)]


 #       boxes_draw =  [[]] + [all_boxes[k][i] for k in range(1, imdb.num_classes)]

#        print(boxes_draw)
        im = draw_all_detection(data['data'],boxes_draw,imagenet_cls_names,scale)
	_mkdir(save_path)
        _mkdir(save_path + "/cls_right")
        _mkdir(save_path + "/det_right")
        _mkdir(save_path + "/det_miss")
       # boxes_this_image = np.array([all_boxes[j][i] for j in range(1, imdb.num_classes)])
        print(gt_rois.shape)
        for j in range(0, len(gt_rois[0])):
            cls_right = 0
            det_right = 0
            det_miss = 0
            gt_roi = gt_rois[0][j]
            if gt_roi.shape[0] == 0: 
                continue
#`            print(boxes_this_image)
            overlap, idx = _get_max_overlap(gt_roi[1:5],np.array(boxes_this_image),scale)
            if overlap > 0.5 :
                print(boxes_this_image[idx])
                if boxes_this_image[idx][5] == roidb[i]['gt_classes'][j]:
                    cls_right =1
                else:
                    det_right = 1
            else:
                det_miss =1
            im_bk = im.copy()
            im_bk = _draw_gtbox(im_bk,gt_roi[1:5],imagenet_cls_names[roidb[i]['gt_classes'][j]],1)
            image_name = os.path.basename(roidb[i]['image']) + "_idx_"+ str(j) + "_label_" + str(roidb[i]['gt_classes'][j]) +".jpg"
            if cls_right:
                savefile = save_path + "/cls_right/" + image_name
            if det_right:
                savefile = save_path + "/det_right/" + image_name
            if det_miss:
                savefile = save_path + "/det_miss/" + image_name
            print(savefile)
            cv2.imwrite(savefile,im_bk)





def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    # general
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    parser.add_argument('--network', help='network name', default=default.network, type=str)
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)
    parser.add_argument('--image_set', help='image_set name', default=default.test_image_set, type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    parser.add_argument('--dets_path', help='dets path', default="", type=str)
    parser.add_argument('--save_path', help='save path', default="", type=str)

    parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    draw_dets(args.dataset, args.image_set, args.root_path, args.dataset_path,args.dets_path,
             args.thresh,args.save_path)

if __name__ == '__main__':
    main()
