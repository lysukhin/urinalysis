import cv2
import numpy as np
import glob
import os
from collections import OrderedDict
import matplotlib.pyplot as plt


def get_path_contents(path):
    return sorted(glob.glob(os.path.join(path,"*")))

def get_path_white(path_illum):
    
    path_whites = []
    for x in path_illum:
        if "white" in os.path.basename(x):
            path_whites.append(x)
            
    return path_whites


def get_colorname(path,char_to_split_name="_"):
    return os.path.basename(path).split(char_to_split_name)[0]


def group_by_colors(image_paths,convert_to_image=False,char_to_split_name="_"):
    
    if not convert_to_image:
        color_paths = OrderedDict()
        for x in image_paths:
            color = get_colorname(x,char_to_split_name)

            if color not in color_paths:
                color_paths[color] = []
                color_paths[color].append(x)
            else:
                color_paths[color].append(x)
    else:
        color_paths = OrderedDict()
        for x in image_paths:
            color = get_colorname(x,char_to_split_name)

            img_bgr = cv2.imread(x,cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
            if color not in color_paths:
                color_paths[color] = []
                color_paths[color].append(img_rgb)
            else:
                color_paths[color].append(img_rgb)
        return color_paths


def pyrDowning(img_rgb,n_downs=2):
    img = img_rgb
    for i in xrange(n_downs):
        img = cv2.pyrDown(img)
        
    return img


def compress_photos(path_to_raw_images,n_downs=3):

    path_dir = os.path.dirname(path_to_raw_images)
    path_compressed = os.path.join(path_dir,"compressed")

    try:
        os.mkdir(path_compressed)
    except OSError:
        print "Directory already exists"

    for path_to_raw_illum_folder in get_path_contents(path_to_raw_images):

        basename = os.path.basename(path_to_raw_illum_folder)
        path_to_new_folder = os.path.join(path_compressed,basename)

        try:    
            os.mkdir(path_to_new_folder)
        except OSError:
            print "Directory already exists"

        for img_path in get_path_contents(path_to_raw_illum_folder):
            img = cv2.imread(img_path)
            img = pyrDowning(img,n_downs=n_downs)

            path_to_compressed_file = os.path.join(path_to_new_folder,
                                                  os.path.basename(img_path))
            cv2.imwrite(path_to_compressed_file,img)
    return path_compressed


class Illumination(object):

    def __init__(self):
        self.image_paths = None
        self.images = None
        self.name = None
        
    def load_images(self,path_illum,char_to_split_name):
        self.name = os.path.basename(path_illum)
        image_paths = get_path_contents(path_illum)
        self.image_paths = group_by_colors(image_paths,convert_to_image=False,char_to_split_name=
                                          char_to_split_name)
        self.images = group_by_colors(image_paths,convert_to_image=True,char_to_split_name=
                                     char_to_split_name)
    

    
class Experiment(object):
    
    def __init__(self):
        self.illuminations = {}
    
    def load_images(self,path,char_to_split_color_name="_"):
        path_illuminations = get_path_contents(path)
        
        for path_illum in path_illuminations:
            I = Illumination()
            I.load_images(path_illum,char_to_split_color_name)
            self.illuminations[I.name] = I

            
def adjust_white_balance_wrt_rgb(image, (r_w, g_w, b_w), white_value=255):
    """
    Take an RGB-image with (r_w, g_w, b_w) being a RGB values of a 'truly white' object on this image,
    then scale all channels in a way that these values become white_value.
    """

    assert len(image.shape) == 3, "Not a 3-channel image"

    image_wb = image.copy()

    r_coeff = 1. * white_value / r_w
    g_coeff = 1. * white_value / g_w
    b_coeff = 1. * white_value / b_w

    image_wb[:, :, 0] = np.minimum(image_wb[:, :, 0] * r_coeff, 255).astype('uint8')
    image_wb[:, :, 1] = np.minimum(image_wb[:, :, 1] * g_coeff, 255).astype('uint8')
    image_wb[:, :, 2] = np.minimum(image_wb[:, :, 2] * b_coeff, 255).astype('uint8')

    return image_wb

def plot_color_hist(img):
    colors = ['r','g','b']
    for i in xrange(3):
            color_values = np.ravel(img[:,:,i])

            mean = np.mean(color_values)
            std = np.std(color_values)
            plt.hist(color_values,color=colors[i],label = "%.1f $ \pm $ %.1f"%(mean,std),bins=20,alpha=0.5)