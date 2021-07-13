# import torch libraries
import os
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor

    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

# import the utility functions
import os
import pylab
import numpy as np
import pandas as pd
from PIL import Image
import skimage.io as io
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import glob
import sys
import getopt
import cv2
import csv
from functools import reduce
import pdb
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import random
import shapefile
from torchvision.utils import save_image
from skimage import io

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class TestDataset(Dataset):
    def __init__(self, fileNames, rootDir, transform=None, target_transform=None):
        self.rootDir = rootDir
        self.transform = transform
        self.targetTransform = target_transform
        self.frame = pd.read_csv(fileNames, dtype=str, delimiter=' ',header=None)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        # input and target images
        fname = self.frame.iloc[idx, 0]
        inputName = os.path.join(self.rootDir, fname)
        targetName = None
        # process the images
        inputImage = Image.open(inputName).convert('RGB')
        # i, j, h, w = transforms.RandomCrop.get_params(inputImage, output_size=(256, 256))
        # inputImage = transforms.functional.crop(inputImage, i, j, h, w)


        if self.transform is not None:
            inputImage = self.transform(inputImage)

        if self.targetTransform is not None:
            targetName = os.path.join(self.rootDir, self.frame.iloc[idx, 1])
            targetImage = Image.open(targetName).convert('L')
            # targetImage = transforms.functional.crop(targetImage, i, j, h, w)
            # targetImage_np = np.array(targetImage)
            # print(np.unique(targetImage_np,return_counts=True))
            targetImage = self.targetTransform(targetImage)
            # save_image(inputImage, 'inputimg/inp{}.jpg'.format(idx))
            # save_image(targetImage, 'inputimg/inp{}.png'.format(idx))
            return inputImage, fname , targetImage

        return inputImage, fname

def weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv2d') != -1:
        init.xavier_uniform(m.weight.data)
        init.constant(m.bias.data, 0.1)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def grayTrans(img):
    img = img.numpy()[0][0] * 255.0
    img = (img).astype(np.uint8)
    return img

def transfer_weights(model_from, model_to):
    wf = copy.deepcopy(model_from.state_dict())
    wt = model_to.state_dict()
    for k in wt.keys():
        if not k in wf:
            wf[k] = wt[k]
    model_to.load_state_dict(wf)


def convert_vgg(vgg16):
    net = vgg()
    vgg_items = list(net.state_dict().items())
    vgg16_items = list(vgg16.items())
    pretrain_model = {}
    j = 0
    for k, v in net.state_dict().items():
        v = vgg16_items[j][1]
        k = vgg_items[j][0]
        pretrain_model[k] = v
        j += 1
    return pretrain_model


class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()

        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=35),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv2
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv3
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv4
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv5
        self.conv5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        return conv5


class HED(nn.Module):
    def __init__(self):
        super(HED, self).__init__()

        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv2
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv3
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv4
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv5
        self.conv5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(128, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn5 = nn.Conv2d(512, 1, 1)
        self.fuse = nn.Conv2d(5, 1, 1)

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.dsn1 = self.dsn1.cuda()
            self.dsn2 = self.dsn2.cuda()
            self.dsn3 = self.dsn3.cuda()
            self.dsn4 = self.dsn4.cuda()
            self.dsn5 = self.dsn5.cuda()
            self.fuse = self.fuse.cuda()

    def forward(self, x):
        h = x.size(2)
        w = x.size(3)

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        # conv5 = self.conv5(conv4)
        conv5 = conv4

        ## side output

        d1 = self.dsn1(conv1)
        d2 = F.upsample_bilinear(self.dsn2(conv2), size=(h, w))
        d3 = F.upsample_bilinear(self.dsn3(conv3), size=(h, w))
        d4 = F.upsample_bilinear(self.dsn4(conv4), size=(h, w))
        d5 = F.upsample_bilinear(self.dsn5(conv5), size=(h, w))

        # print('dsn1 shape : ', self.dsn1(conv1).shape)
        # print('dsn2 shape : ', self.dsn2(conv2).shape)
        # print('dsn3 shape : ', self.dsn3(conv3).shape)
        # print('dsn4 shape : ', self.dsn4(conv4).shape)
        # print('dsn5 shape : ', self.dsn5(conv5).shape)

        # dsn fusion output
        fuse = self.fuse(torch.cat((d1, d2, d3, d4, d5), 1))

        d1 = F.sigmoid(d1)
        d2 = F.sigmoid(d2)
        d3 = F.sigmoid(d3)
        d4 = F.sigmoid(d4)
        d5 = F.sigmoid(d5)
        fuse = F.sigmoid(fuse)

        return d1, d2, d3, d4, d5, fuse

def plotResults2(inp, images, gt, size, fname, thres, image_path, evaluation, token):
    images[0] = images[0].astype(int)
    tmp = image_path.split("media")
    segmentation_area = 1000
    images[0][images[0] < thres] = 0
    images[0][images[0] >= thres] = 255
    pylab.rcParams['figure.figsize'] = size, size
    gts = 2
    plt.axis('off')
    # plt.imshow(images[0])
    # plt.savefig(tmp[0].split('media')[0] + 'media' + tmp[0].split('media')[1][:9] + '//%d//HED//mask.png' % token,
    #             bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.clf()
    if evaluation == False:
        gts = 1
    for i in range(0, gts):
        if i == 0:
            plt.imshow(inp)
            # plt.imshow(inp)
        if i == 1:
            plt.imshow(gt, cmap=cm.Greys_r)
            # plt.imshow(gt)

    im = Image.open(image_path)
    im.save(tmp[0] + 'media' + tmp[1][:10] +'%d/HED/Image.jpg'%token)
    io.imsave(tmp[0] + 'media' + tmp[1][:10] + '%d/HED/mask.png'%token, images[0])


    img_contour = images[0].astype(np.uint8)
    contours, hierachy = cv2.findContours(img_contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area_list = []
    area_ind = []
    k=0
    for j in range(len(contours)):
        if len(contours[j]) > 1 and cv2.contourArea(contours[j]) > segmentation_area:
            area_list.append(int(cv2.contourArea(contours[j])))
            x0, y0 = zip(*np.squeeze(contours[j]))
            x0_mean =np.mean(x0)
            y0_mean =np.mean(y0)
            k +=1
            polygon = []
            for m in range(len(x0)):
                polygon.append([x0[m], -y0[m]])
            w = shapefile.Writer(tmp[0] + 'media' + tmp[1][:10] + '%d/HED/polygon_%d'%(token,k))
            w.field('region', 'N')
            w.field('area', 'N')
            w.poly([polygon])
            w.record(region=k, area=int(cv2.contourArea(contours[j])))
            w.close
            plt.plot(x0, y0, c="b", linewidth=1.0)
            plt.plot(x0_mean, y0_mean, marker='$%d$'%k, markersize = 10)
            area_ind.append(k)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(tmp[0] + 'media' + tmp[1][:10] + '%d/HED/contour.png'%token, bbox_inches='tight', dpi=400, pad_inches=0)

    h = inp.shape[0]
    w = inp.shape[1]
    zeros = np.zeros((h, w))
    y_pred = images[0]
    ones = y_pred.reshape(h, w)
    mask = np.stack((ones, zeros, zeros, ones), axis=-1)
    
    plt.imshow(mask, alpha=0.3)
    plt.axis('off')
    plt.savefig(tmp[0] + 'media' + tmp[1][:10] + '%d/HED/maskContour.png'%token, dpi=400, pad_inches=0, bbox_inches='tight')
    plt.clf()

    plt.imshow(inp)
    plt.imshow(mask, alpha=0.3)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(tmp[0] + 'media' + tmp[1][:10] + '%d/HED/mask_overlay.png'%token, pad_inches=0, bbox_inches='tight', dpi=400)
    plt.clf()

    results = {
        'area_ind': area_ind,
        'area': area_list
    }
    result_hed = results
    return result_hed

class Segmentation_HED:
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        self.result_hed = None
        self.MODEL_NAME = 'HED0_v3_retrain.pth'
        # self.MODEL_NAME = 'HED0_v4_retrain.pth'
        # self.MODEL_NAME = 'seg_model_v4.h5'
        # self.MODEL_NAME = 'seg_check_v3130-0.16.h5'


        self.arg_threshold = 120


    def union_intersect(true, pred, threshold=100):
        # Predict matrix, GT matrix vectorize for Intersection 1d , Union 1d, setDiff 1d Calculation
        h, w = true.shape
        nflat = true.ravel().shape

        pred = pred.copy()
        true = true.copy()

        pred = pred.astype(int)
        true = true.astype(int)

        pred[pred < threshold] = 0
        pred[pred >= threshold] = 255
        true_ravel = true.ravel()
        pred_ravel = pred.ravel()

        # Find index 255. or 1. region
        true_ind = np.where(true_ravel == 1)
        pred_ind = np.where(pred_ravel == 255)

        # Intersection , Union , Diff Calculation
        TP_ind = np.intersect1d(true_ind, pred_ind)
        FN_ind = np.setdiff1d(true_ind, TP_ind)
        FP_ind = np.setdiff1d(pred_ind, TP_ind)
        union_ind = reduce(np.union1d, (TP_ind, FN_ind, FP_ind))

        # Intersection of Union(HED,GT)

        TP_count = TP_ind.shape[0]
        union_count = union_ind.shape[0]
        pred_count = pred_ind[0].shape[0]
        true_count = true_ind[0].shape[0]

        precision = 0
        iou = 0
        recall = 0
        f1 = 0
        print('THRES({}) - TP : {}, UNION : {}, PRED : {}, TRUE : {}'.format(threshold, TP_count, union_count, pred_count,
                                                                             true_count))
        if TP_count == 0 or pred_count == 0 or true_count == 0 or union_count == 0:
            pass

        else:
            iou = TP_count / union_count
            precision = TP_count / pred_count
            recall = TP_count / true_count
            print(precision, recall)

            f1 = 2 * (precision * recall) / (precision + recall)

        # Create dummy array
        union = np.zeros(nflat)
        TP = np.zeros(nflat)
        FN = np.zeros(nflat)
        FP = np.zeros(nflat)

        # Write Array
        union[union_ind] = 255
        TP[TP_ind] = 255
        FN[FN_ind] = 255
        FP[FP_ind] = 255

        # return 2d arrays and iou
        return np.reshape(union, true.shape), np.reshape(TP, true.shape), np.reshape(FP, true.shape), np.reshape(FN,
                                                                                                                 true.shape), precision, recall, iou, f1


    def plotResults(inp, images, gt, size, fname, thres):
        pylab.rcParams['figure.figsize'] = size, size
        gts = 4
        loss_is = 3

        prev_union, prev_TP, prev_FP, prev_FN, prev_precision, prev_recall, prev_iou, prev_f1 = union_intersect(gt,
                                                                                                                images[0],
                                                                                                                threshold=thres - 20)
        union, TP, FP, FN, precision, recall, iou, f1 = union_intersect(gt, images[0], threshold=thres)
        next_union, next_TP, next_FP, next_FN, next_precision, next_recall, next_iou, next_f1 = union_intersect(gt,
                                                                                                                images[0],
                                                                                                                threshold=thres + 20)

        if evaluation == False:
            gts = 1
            loss_is = 0

        for i in range(0, loss_is):
            s = plt.subplot(3, 5, i + 1)
            if i == 0:
                title = '{} F1 : {}'.format(thres - 20, round(prev_f1, 3))
                plt.imshow(np.dstack((prev_TP + prev_FP, prev_FN + prev_FP, np.zeros(TP.shape))))
            if i == 1:
                title = '{} F1 : {}'.format(thres, round(f1, 3))
                plt.imshow(np.dstack((TP + FP, FN + FP, np.zeros(TP.shape))))
                img = np.dstack((TP + FP, FN + FP, np.zeros(TP.shape)))
                b, g, r = cv2.split(img)
                img2 = cv2.merge([r, g, b])
                y_pred = TP + FP

                h = inp.shape[0]
                w = inp.shape[1]
                zeros = np.zeros((h, w))
                y_pred = TP + FP
                ones = y_pred.reshape(h, w)
                mask = np.stack((ones, zeros, zeros, ones), axis=-1)

                inp_reshape = Image.open('data\\bd_vh_02\\' + fname[0]).convert('RGBA')
                img3 = inp_reshape + mask
                b2, g2, r2, a2 = cv2.split(img3)
                img4 = cv2.merge([r2, g2, b2, a2])
                cv2.imwrite(os.path.join('output', fname[0][16:]), img2)
                cv2.imwrite(os.path.join('output', fname[0][16:-4]) + '.png', TP + FP)
                cv2.imwrite(os.path.join('output', fname[0][16:-4]) + '_overlay.jpg', img4)

            if i == 2:
                title = '{} F1 : {}'.format(thres + 20, round(next_f1, 3))
                plt.imshow(np.dstack((next_TP + next_FP, next_FN + next_FP, np.zeros(TP.shape))))

            s.set_xticklabels([])
            s.set_yticklabels([])
            s.yaxis.set_ticks_position('none')
            s.xaxis.set_ticks_position('none')
            s.set_title(title, fontsize=35)

        titles = ['INPUT', 'GT', 'UNION(GT,HED)', 'INTER(GT,HED)', 'HED', 'S1', 'S2', 'S3', 'S4']
        for i in range(0, gts):
            s = plt.subplot(3, 5, i + 6)
            if i == 0:
                plt.imshow(inp, cmap=cm.Greys_r)
                # plt.imshow(inp)
            if i == 1:
                plt.imshow(gt.astype(np.uint8), cmap=cm.Greys_r)
                # plt.imshow(gt)
            if i == 2:
                plt.imshow(union, cmap=cm.Greys_r)
                # plt.imshow(union)
            if i == 3:
                plt.imshow(TP, cmap=cm.Greys_r)
                # plt.imshow(TP)
            s.set_xticklabels([])
            s.set_yticklabels([])
            s.yaxis.set_ticks_position('none')
            s.xaxis.set_ticks_position('none')
            s.set_title(titles[i], fontsize=35)

        for i in range(0, len(images)):
            s = plt.subplot(3, 5, i + 11)
            plt.imshow(images[i], cmap=cm.Greys_r)
            # plt.imshow(images[i])
            s.set_xticklabels([])
            s.set_yticklabels([])
            s.yaxis.set_ticks_position('none')
            s.xaxis.set_ticks_position('none')
            s.set_title(titles[i + 4], fontsize=35)
        plt.tight_layout()

        if 'croppedimg' in fname[0].split('/'):
            plt.savefig(os.path.join('output', fname[0][16:].replace('jpg', 'jpeg')))
        else:
            # plt.savefig(os.path.join('output',fname[0].split('/')[2].replace('jpg','jpeg')))
            plt.savefig(os.path.join('output', fname[0][16:].replace('jpg', 'jpeg')))

        return fname[0], thres, iou, precision, recall, f1



    def plot_contour_overlay(inp, images, gt, size, fname, thres):
        union, TP, FP, FN, precision, recall, iou, f1 = union_intersect(gt, images[0], threshold=thres)

        y_pred = TP + FP

        img_contour = y_pred.astype(np.uint8)
        contours, hierachy = cv2.findContours(img_contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        area_list = []
        for i in range(len(contours)):
            if len(contours[i]) > 1 and cv2.contourArea(contours[i]) > 200:
                area_list.append(cv2.contourArea(contours[i]))
                x0, y0 = zip(*np.squeeze(contours[i]))
                plt.plot(x0, y0, c="b", linewidth=1.0)

        h = inp.shape[0]
        w = inp.shape[1]
        zeros = np.zeros((h, w))
        y_pred = TP + FP
        ones = y_pred.reshape(h, w)
        mask = np.stack((ones, zeros, zeros, ones), axis=-1)
        plt.imshow(inp)
        plt.imshow(mask, alpha=0.3)
        plt.xlim(0, w)
        plt.ylim(h, 0)

        plt.savefig(os.path.join('output', fname[0][16:-4]) + '_contour_overlay.jpg', dpi=300)
        plt.clf()
        return fname[0], thres, iou, precision, recall, f1

    def inference_by_path(self, image_path, token):

        tmp = image_path.split("media")
        nVisualize = 1
        inp = None
        fname = None
        gt = None
        input_img = None
        gt_img = None

        # make directory
        if not (os.path.isdir(tmp[0] + 'media' + tmp[1][:10] +'%d'%token)):
            os.makedirs(os.path.join(tmp[0] +'media' + tmp[1][:10] +'%d'%token))
        if not (os.path.isdir(tmp[0] + 'media' + tmp[1][:10] + '%d/HED'%token)):
            os.makedirs(os.path.join(tmp[0] + 'media' + tmp[1][:10] + '%d/HED'%token))

        arg_Model = os.path.join(self.path, self.MODEL_NAME)
        arg_DataRoot = tmp[0] + 'media' + tmp[1][:9]
        arg_Thres = self.arg_threshold
        print(arg_Model, arg_DataRoot, arg_Thres)

        for Opt, Arg in getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
            if Opt == '--model' and Arg != '': arg_Model = Arg
            if Opt == '--data' and Arg != '': arg_DataRoot = Arg
            if Opt == '--thres' and Arg != '': arg_Thres = float(Arg)

        # using evaluation metrics(Must have data in the croppedgt directory)
        evaluation = False

        # create instance of HED model
        net = HED()
        net.cuda()
        # device = torch.device('cpu')
        # load the weights for the model
        net.load_state_dict(torch.load(arg_Model))

        # batch size
        nBatch = 1

        # make test list for infer
        img_paths = glob.glob(os.path.join(arg_DataRoot, tmp[1][10:]))
        txtfile = open(os.path.join(arg_DataRoot, 'test.lst'), 'w')
        for img_path in img_paths:
            saved_img = os.path.relpath(img_path, arg_DataRoot)
            txtfile.write('{} \n'.format(saved_img))
        txtfile.close()

        # create data loaders from dataset
        testPath = os.path.join(arg_DataRoot, 'test.lst')
        print(testPath)

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        targetTransform = transforms.Compose([
            transforms.ToTensor()
        ])

        testDataset = None
        if evaluation:
            testDataset = TestDataset(testPath, arg_DataRoot, transform, targetTransform)
        else:
            testDataset = TestDataset(testPath, arg_DataRoot, transform)
        testDataloader = DataLoader(testDataset, batch_size=nBatch)

        for i, sample in enumerate(testDataloader):
            # get input sample image
            if evaluation:
                print(fname)
                inp, fname, gt = sample
                gt = Variable(gt)
                file_name = fname[0]
            else:
                inp, fname = sample
            inp = Variable(inp.cuda())

            iou, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
            # perform forward computation
            s1, s2, s3, s4, s5, s6 = net.forward(inp)

            # convert back to numpy arrays

            out = []
            out.append(grayTrans(s6.data.cpu()))
            out.append(grayTrans(s1.data.cpu()))
            out.append(grayTrans(s2.data.cpu()))
            out.append(grayTrans(s3.data.cpu()))
            out.append(grayTrans(s4.data.cpu()))

            inp2 = inp.data[0].permute(1, 2, 0)

            input_img = inp2.cpu().numpy()
            # input_img = inp.data[0].cpu().numpy()[0]
            if evaluation:
                img = Image.fromarray(out[0], 'L')
                # img.save(os.path.join('output/pred', fname[0].split('/')[2].replace('jpg', 'jpeg')))
                img.save(os.path.join('output/pred', fname[0][16:].replace('jpg', 'jpeg')))
                gt_img = gt.data[0].cpu().numpy()[0]
                print(gt_img.shape)

            # visualize every 10th image
            if i % nVisualize == 0:
                # if (len(gt_unique[0])==2 and gt_unique[1][1]>1000): # unique value is not 0.
                #     print(len(gt_unique[0]),gt_unique[1][1])
                if evaluation:
                    file_name, thres, iou, precision, recall, f1 = plotResults(input_img, out, gt_img, 25, fname, arg_Thres)
                    # file_name, thres, iou, precision, recall, f1=plot_contour_overlay(input_img, out, gt_img, 25, fname, arg_Thres)
                    f = open(os.path.join('output', 'output.csv'), 'a', encoding='utf-8', newline='')
                    wr = csv.writer(f)
                    wr.writerow([file_name, thres, iou, precision, recall, f1])
                    f.close()

                else:
                    result_hed = plotResults2(input_img, out, gt_img, 25, fname, arg_Thres, image_path, evaluation, token)

        self.result_hed = result_hed
        return self.result_hed