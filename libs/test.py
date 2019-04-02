import cv2
from colorizer.tf.utils import *
from colorizer.tf.net import Net
from skimage.io import imsave
from Akhil.settings import BASE_DIR


def colorize():
    """
    This method reads in a black and white image and converts
    it to colour image.
    """
    # Read the B/W image.

    img = cv2.imread(BASE_DIR + '/colorizer/static/colorizer/gray.jpg')
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[None, :, :, None]

    data_l = (img.astype(dtype=np.float32)) / 255.0 * 100 - 50

    # As we are testing train should be false.
    autocolor = Net(train=False)

    # encode returns a loaded_model
    model = autocolor.encode(data_l)
    # decode returns a RGB image
    img_rgb = decode(data_l, model)

    # Save the Colour image.
    imsave('colorizer/static/colorizer/color.jpg', img_rgb)
