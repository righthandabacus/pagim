"""Image processing function"""

import cv2
import numpy as np


# Register image processing functions
opdict = {}


def image_op(opname):
    """Function decorator to register image operations.
    All image processing functions should be decorated with this. The function
    should take an image as input, return an output image and a code (str or
    list of str).
    """
    def _deco(fn):
        opdict[opname] = fn
        return fn
    return _deco


#
#
# Decorated functions below, in order of registration to the opdict
#
#


# Color conversion

@image_op("Grayscale")
def fliplr(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img, "img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)"


@image_op("RGB->YCrCb get Y")
def ycrcb_y(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, _, _ = cv2.split(img)
    return y, "img = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb))[0]"


@image_op("RGB->YUV get Y")
def yuv_y(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y, u, v = cv2.split(yuv)
    return y, "img = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2YUV))[0]"


@image_op("RGB->YUV get U")
def yuv_u(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y, u, v = cv2.split(yuv)
    b = np.zeros_like(u)
    g = 255 - u
    rgb = cv2.merge([u,g,b])
    code = [
        "u = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2YUV))[1]",
        "img = cv2.merge([u, 255-u, np.zeros_like(u)])"
    ]
    return rgb, code


@image_op("RGB->YUV get V")
def yuv_v(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y, u, v = cv2.split(yuv)
    r = np.zeros_like(v)
    g = 255 - v
    rgb = cv2.merge([r,g,v])
    code = [
        "v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2YUV))[-1]",
        "img = cv2.merge([np.zeros_like(v), 255-v, v])"
    ]
    return rgb, code



# Rotation and flips

@image_op("Horizontal flip")
def mirror(img):
    img = cv2.flip(img, 1)
    return img, "img = cv2.flip(img, 1)"


@image_op("Vertical flip")
def flip(img):
    img = cv2.flip(img, 0)
    return img, "img = cv2.flip(img, 0)"


@image_op("Rotate 180 deg")
def rot180(img):
    # alt.: img = cv2.rotate(img, cv2.ROTATE_180)
    img = cv2.flip(img, -1)
    return img, "img = cv2.flip(img, -1)"


@image_op("Rotate 90 deg clockwise")
def rot270(img):
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img, "img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)"


@image_op("Rotate 90 deg counterclockwise")
def rot90(img):
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img, "img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)"


# Feature extraction

@image_op("Binarize a grayscale image")
def binarize(img):
    "Binarize with threshold 127"
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    code = [
        "_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)"
    ]
    return img, code


@image_op("Dilate a binarized image")
def dilate(img):
    # equiv.: kernel = np.ones((5, 5), np.uint8)
    # MORPH_RECT is a full matrix, MORPH_ELLIPSE is circle-like, MORPH_CROSS is cross-like
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    img = cv2.dilate(img, kernel, iterations=1)
    code = [
        "kernel = np.ones((5, 5), np.uint8)",
        "img = cv2.dilate(img, kernel, iterations=1)"
    ]
    return img, code


@image_op("Erode a binarized image")
def erode(img):
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    code = [
        "kernel = np.ones((5, 5), np.uint8)",
        "img = cv2.erode(img, kernel, iterations=1)"
    ]
    return img, code


@image_op("Gradient a binarized image (diff between erode and dilate)")
def gradient(img):
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    code = [
        "kernel = np.ones((5, 5), np.uint8)",
        "img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)"
    ]
    return img, code


@image_op("Opening a binarized image: Erode then dilate")
def opening(img):
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    code = [
        "kernel = np.ones((5, 5), np.uint8)",
        "img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)"
    ]
    return img, code


@image_op("Closing a binarized image: Dilate then erode")
def closing(img):
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    code = [
        "kernel = np.ones((5, 5), np.uint8)",
        "img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)"
    ]
    return img, code


# Distort

@image_op("Gaussian blur (radius 11)")
def camera_blur(img):
    img = cv2.GaussianBlur(img, (11,11), 0)
    return img, "img = cv2.GaussianBlur(img, (11,11), 0)"


@image_op("Reshape")
def reshape(img):
    img = cv2.resize(img, (int(img.shape[1] * 1.2), img.shape[0]))
    return img,"img = cv2.resize(img, (int(img.shape[1] * 1.2), img.shape[0]))"


@image_op("Add Gaussian noise")
def add_noise(img):
    # Gaussian additive noise with fixed variance
    mean = 0
    std_dev = 0.1
    noise = np.random.normal(mean, std_dev, img.shape)
    img = (img + noise).astype(np.uint8)
    code = "img = (img + np.random.normal(0, 0.1, img.shape)).astype(np.uint8)"
    return img, code


@image_op("Add multiplicative noise")
def add_multiplicative_noise(img):
    # Gaussian multiplicative noise
    # see also https://scikit-image.org/docs/dev/api/skimage.util.html#skimage.util.random_noise
    noise = img * np.random.randn(*(img.shape))
    img = (img + noise).astype(np.uint8)
    code = "img = (img + img * np.random.randn(*(img.shape))).astype(np.uint8)"
    return img, code


@image_op("Add salt & pepper noise")
def add_saltpepper(img):
    sp = 0.5  # salt to pepper ratio
    amount = 0.005  # amount of values changed in total
    # add salt: Set some element to 255
    n_salt = int(img.size * sp * amount)
    coords = [np.random.randint(0, n-1, n_salt) for n in img.shape]
    img[coords] = 255
    # add pepper: Set some element to 0
    n_pepper = int(img.size * (1-sp) * amount)
    coords = [np.random.randint(0, n-1, n_pepper) for n in img.shape]
    img[coords] = 0

    code = [
        f"coords = [np.random.randint(0, n-1, {n_salt}) for n in img.shape]",
        "img[coords] = 255",
        f"coords = [np.random.randint(0, n-1, {n_pepper}) for n in img.shape]",
        "img[coords] = 0"
    ]
    return img


@image_op("Canny edge on grayscale image")
def canny_edge(img):
    # blur to remove small noise first
    blur = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.Canny(blur, 75, 200)
    return img, "img = cv2.Canny(cv2.GaussianBlur(img, (5,5), 0), 75, 200)"


@image_op("Sobel edge on grayscale image")
def sobel_edge(img):
    # blur to remove small noise first
    blur = cv2.GaussianBlur(img, (5,5), 0)
    # Find horizontal and vertical gradients using Sobel kernel
    grad_x = cv.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grads = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5)  # weighted average
    img = cv2.convertScaleAbs(grads)   # transform back to uint8
    code = [
        "blur = cv2.GaussianBlur(gray, (5,5), 0)",
        "grad_x = cv.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)",
        "grad_y = cv.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)",
        "img = cv2.convertScaleAbs(cv2.addWeighted(grad_x, 0.5, grad_y, 0.5))"
    ]
    return img, code
