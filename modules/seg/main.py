import numpy as np
import matplotlib
from keras.backend import set_session
import matplotlib.pyplot as plt
import glob
import os
import sys
from PIL import Image
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras_unet.utils import get_augmented
from keras_unet.utils import plot_imgs
from keras_unet.models import custom_unet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import jaccard_distance
from keras_unet.utils import plot_segm_history
import skimage
from skimage import io
import cv2
import tensorflow as tf
import keras
import shapefile

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class Segmentation:
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        self.MODEL_NAME = 'seg_model_v3_1024.h5'
        # self.MODEL_NAME = 'seg_model_v3-1.h5'
        # self.MODEL_NAME = 'seg_model_v4.h5'
        # self.MODEL_NAME = 'seg_check_v3130-0.16.h5'

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        # config.gpu_options.allow_growth = True
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

        self.segmentation_area = 1000
        self.severity_threshold = 0.5
        self.resize_value = 1024
        self.resize_value_2 = 512

    def inference_by_path(self, image_path, token):

        tmp = image_path.split("media")

        imgs_test_list = []
        imgs_test_list.append(np.array(Image.open(image_path).resize((self.resize_value, self.resize_value))))
        imgs_test_np = np.asarray(imgs_test_list)
        x_test = np.asarray(imgs_test_np, dtype=np.float32) / 255

        model = tf.keras.models.load_model(os.path.join(self.path, self.MODEL_NAME),
                                           custom_objects={"iou": iou, "iou_thresholded": iou_thresholded})
        model.load_weights(os.path.join(self.path, self.MODEL_NAME))
        y_pred = model.predict(x_test)

        # Change prediction value float 32 to uint8(from float[0-1] to integer[0-255])
        for i in range(self.resize_value):
            for j in range(self.resize_value):
                if y_pred[0, i, j] > self.severity_threshold:
                    y_pred[0, i, j] = 255

                else:
                    y_pred[0, i, j] = 0

        y_pred = y_pred.astype(np.uint8)

        # mask image
        mask_ = y_pred[0, :, :, 0]
        mask_mod = Image.fromarray(mask_)

        # make directory
        if not (os.path.isdir(tmp[0] + 'media' + tmp[1][:10] +'%d'%token)):
            os.makedirs(os.path.join(tmp[0] + 'media' + tmp[1][:10] + '%d'%token))
        if not (os.path.isdir(tmp[0] + 'media' + tmp[1][:10] + '%d/UNET'%token)):
            os.makedirs(os.path.join(tmp[0] + 'media' + tmp[1][:10] + '%d/UNET'%token))
        # vectorizing through contour method and calculate mask area

        img = y_pred[0].astype(np.uint8)
        ret, thresh = cv2.threshold(img[:, :, 0], 127, 255, 0)
        contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        area_list = []
        area_ind = []
        k = 0
        for i in range(len(contours)):
            if len(contours[i]) > 1 and cv2.contourArea(contours[i]) > self.segmentation_area:
                area_list.append(int(cv2.contourArea(contours[i])))
                x0, y0 = zip(*np.squeeze(contours[i]))
                x0_mean =np.mean(x0)
                y0_mean =np.mean(y0)
                k += 1
                polygon = []
                for m in range(len(x0)):
                    polygon.append([x0[m], -y0[m]])
                w = shapefile.Writer(tmp[0] + 'media' + tmp[1][:10] + '%d/UNET/polygon_%d'%(token,k))
                w.field('region', 'N')
                w.field('area', 'N')
                w.poly([polygon])
                w.record(region=k, area=int(cv2.contourArea(contours[i])))
                w.close
                plt.plot(x0, y0, c="b", linewidth=1.0)
                plt.plot(x0_mean, y0_mean, marker='$%d$'%k, markersize = 10)
                area_ind.append(k)

        h = self.resize_value
        w = self.resize_value
        zeros = np.zeros((h, w))
        ones = y_pred[0].reshape(h, w)
        mask = np.stack((ones, zeros, zeros, ones), axis=-1)
        
        plt.imshow(imgs_test_np[0, :, :])
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()        
        plt.savefig(tmp[0] + 'media' + tmp[1][:10] + '%d/UNET/contour.png'%token, bbox_inches='tight', pad_inches=0, dpi=400)
        plt.imshow(mask, alpha=0.3)
        # io.imsave(save_dir[j], imgs_test_np[j,:,:]+mask)
        # plt.imshow(img[:,:,0])

        # plt.xlim(0, self.resize_value)
        # plt.ylim(self.resize_value, 0)

        # plt.show()
        im = Image.open(image_path)

        # plt.savefig(tmp[0].split('.jpg')[0] + '_%d'%token, dpi=300)
        
        im.save(tmp[0] + 'media' + tmp[1][:10] + '%d/UNET/Image.jpg'%token)
        mask_mod.save(tmp[0] + 'media' + tmp[1][:10] + '%d/UNET/mask.png'%token)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(tmp[0] + 'media' + tmp[1][:10] + '%d/UNET/maskContour.png'%token, bbox_inches='tight', pad_inches=0, dpi=400)
        plt.clf()

        plt.imshow(imgs_test_np[0, :, :])
        plt.imshow(mask, alpha=0.3)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(tmp[0] + 'media' + tmp[1][:10] + '%d/UNET/mask_overlay.png'%token, bbox_inches='tight', pad_inches=0, dpi=400)
        plt.clf()

        results = {
                  'area_ind': area_ind,
                  'area': area_list
                  }
        self.result = results
        return self.result


