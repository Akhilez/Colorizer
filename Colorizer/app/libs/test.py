import tensorflow as tf
import cv2
from app.tf.utils import *
from app.tf.net import Net
from skimage.io import imsave
from skimage.transform import resize


def colorize():
    """
    This method reads in a black and white image and converts
    it to colour image.
    """
    # Read the B/W image.
    img = cv2.imread('static/gray.jpg')
    if len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img[None, :, :, None]

    data_l = (img.astype(dtype=np.float32)) / 255.0 * 100 - 50

    # As we are testing train should be false.
    autocolor = Net(train=False)

    # encode returns a loaded_model
    model = autocolor.encode(data_l)
    # decode returns a RGB image
    img_rgb = decode(data_l, model)

    # Save the Colour image.
    imsave('static/color.jpg', img_rgb)
