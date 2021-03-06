import argparse
import pprint
import mxnet as mx

from ..config import config, default, generate_config
from ..symbol import *
from ..dataset import *
from ..core.loader import TestLoader
from ..core.tester import Predictor, generate_proposals
from ..utils.load_model import load_param
import os
import cPickle

def test_rois(network, dataset,pklfile, image_set, root_path, dataset_path,
             ctx, prefix, epoch,
             vis, shuffle, thresh):
    # rpn generate proposal config
    config.TEST.HAS_RPN = True

    # print config
    pprint.pprint(config)

    # load dataset and prepare imdb for training
    imdb = eval(dataset)(image_set, root_path, dataset_path)
    roidb = imdb.gt_roidb()

    # start testing
    if os.path.exists(pklfile):
        with open(pklfile, 'rb') as fid:
            imdb_boxes = cPickle.load(fid)
        print('roidb loaded from {}'.format(pklfile))
        imdb.evaluate_recall(roidb, candidate_boxes=imdb_boxes)   
        return 
    else:
        print("nofile in " + pklfile)
   # imdb_boxes = generate_proposals(predictor, test_data, imdb, vis=vis, thresh=thresh)
   # imdb.evaluate_recall(roidb, candidate_boxes=imdb_boxes)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Region Proposal Network')
    # general
    parser.add_argument('--pklfile', help='input rois file', default="", type=str)
    parser.add_argument('--network', help='network name', default=default.network, type=str)
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)
    parser.add_argument('--image_set', help='image_set name', default=default.test_image_set, type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    # testing
    parser.add_argument('--prefix', help='model to test with', default=default.rpn_prefix, type=str)
    parser.add_argument('--epoch', help='model to test with', default=default.rpn_epoch, type=int)
    # rpn
    parser.add_argument('--gpu', help='GPU device to test with', default=0, type=int)
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--thresh', help='rpn proposal threshold', default=0, type=float)
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print 'Called with argument:', args
    ctx = mx.gpu(args.gpu)
    test_rois(args.network, args.dataset, args.pklfile,args.image_set, args.root_path, args.dataset_path,
             ctx, args.prefix, args.epoch,
             args.vis, args.shuffle, args.thresh)

if __name__ == '__main__':
    main()
