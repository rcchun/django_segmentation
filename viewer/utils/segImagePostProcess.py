import cv2
import numpy as np
import random

def save_image_hightlight_region(seg_img_path, seg_img_hl_path, region_results, patch_size, image_height, image_width) :
    try :
        seg_img = cv2.imread(seg_img_path)
        seg_img_hl = np.zeros([image_height, image_width, 3], dtype=np.uint8)
        for region in region_results:
            region_area = region['region_area']
            b = random.randrange(0, 255)
            g = random.randrange(0, 255)
            r = random.randrange(0, 255)
            for patch in region_area:
                x = patch['x']
                y = patch['y']
                patch = seg_img[y:y + patch_size, x:x + patch_size]
                patch[np.where((patch != [0, 0, 0]).all(axis=2))] = [b, g, r]
                seg_img_hl[y:y + patch_size, x:x + patch_size] = patch

        tmp = cv2.cvtColor(seg_img_hl, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(tmp, 0, 127, cv2.THRESH_BINARY)
        b, g, r = cv2.split(seg_img_hl)
        rgba = [b, g, r, alpha]
        seg_img_hl = cv2.merge(rgba, 4)
        cv2.imwrite(seg_img_hl_path, seg_img_hl)
        return True
    except :
        return False
