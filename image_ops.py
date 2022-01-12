from math import sqrt, ceil
from PIL import Image
import numpy as np

def center_crop(image, crop_size=None):
    """Center crop image.

    Args:
        image: PIL image
        crop_size:  if specified, size of square to center crop
                    otherwise, fit largest square to center of image

    Returns:
        cropped PIL image
    """
    width, height = image.size

    # if crop size not specified, use the largest square in the center of image
    if crop_size is None:
        crop_size = min(height, width)

    # compute crop parameters
    top = int(round((height - crop_size)/2.))
    left = int(round((width - crop_size)/2.))
    bottom = top + crop_size
    right = left + crop_size

    return image.crop((left, top, right, bottom))

def get_image(image_path, image_size_wight, image_size_height, is_crop=False):
    """Load image from file and crop/resize as necessary.

    Args:
        image_path: path to image
        image_size: width/height to resize image
        crop: center crop if True [True]

    Returns:
        numpy array of loaded/cropped/resized image
    """
    # load image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # center crop
    if is_crop:
        img_center_crop = center_crop(img)
    else:
        img_center_crop = img

    # resize
    img_resized = img_center_crop.resize((image_size_height, image_size_wight), Image.ANTIALIAS)

    # convert to numpy and normalize
    img_array = np.asarray(img_resized).astype(np.float32)/127.5 - 1.
    img.close()

    return img_array

def save_image(image, image_path):
    """Save an image.

    Postconditions:
        saves to image file

    Args:
        images: one image
        image_path: path to save image
    """
    # transform back from [-1,1] to [0,1]
    img = (image+1.)/2.

    # save image
    im = Image.fromarray(np.uint8(img*255))
    im.save(image_path)

def save_images(images, image_path):
    """Save images.

    Postconditions:
        saves to image file

    Args:
        images: list of images
        image_path: path to save image
    """
    # transform back from [-1,1] to [0,1]
    images = (images+1.)/2.

    # determine tiled image shape
    num_cols = int(ceil(sqrt(len(images))))
    num_rows = int(ceil(1.*len(images)/num_cols))

    # create new tiled image
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * num_rows, w * num_cols, 3))
    for i, image in enumerate(images):
        c = i % num_cols
        r = i // num_cols
        img[r*h:r*h+h, c*w:c*w+w, :] = image

    # save image
    im = Image.fromarray(np.uint8(img*255))
    im.save(image_path)
