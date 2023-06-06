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
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img, "img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)"


@image_op("Grayscale->RGB")
def gray2rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img, "img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)"


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


@image_op("Brighten by HSV")
def brighten_hsv(img):
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    v = np.clip(v+30, 0, 255)
    img = cv2.cvtColor(cv2.merge([h,s,v]), cv2.COLOR_HSV2RGB)
    code = [
        "h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))",
        "img = cv2.cvtColor(cv2.merge([h,s,np.clip(v+30, 0, 255)]), cv2.COLOR_HSV2RGB)"
    ]
    return img, code


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
def gaussian_blur(img):
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
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grads = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0.0)  # simple weighted average
    img = cv2.convertScaleAbs(grads)   # transform back to uint8
    code = [
        "blur = cv2.GaussianBlur(gray, (5,5), 0)",
        "grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)",
        "grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)",
        "img = cv2.convertScaleAbs(cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0.0))"
    ]
    return img, code


# Sharpening

@image_op("Unsharp mask")
def usm(img):
    radius = 2.0
    weight = -1.0  # USM needs negative weight
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    img = cv2.addWeighted(img, 1-weight, blur, weight, 0).astype(np.uint8)
    code = "img = cv2.addWeighted(img, 2.0, cv2.GaussianBlur(img, (0,0), 2.0), -1.0, 0).astype(np.uint8)"
    return img, code


@image_op("Kernel sharpening")
def kernel_sharpening(img):
    kern = np.array([[ 0,-1, 0],
                     [-1, 5,-1],
                     [ 0,-1, 0]])
    img = cv2.filter2D(img, -1, kern)
    return img, "img = cv2.filter2D(img, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))"


@image_op("Deskew page using Canny edge")
def deskew_canny(img):
    """Complete workflow from RGB image to deskewed"""
    # blur and thresholding
    blur = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (3,3), 2)
    blur = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    blur = cv2.fastNlMeansDenoising(blur, 11, 31, 9)
    # canny edge detection
    edged = cv2.Canny(blur, 50, 150, apertureSize=7)
    # find max contour
    contours, _hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    def _simplify_contour(c):
        hull = cv2.convexHull(c)
        return cv2.approxPolyDP(hull, 0.03*cv2.arcLength(hull, True), True)
    contours = [_simplify_contour(c) for c in contours]
    max_contour = max(contours, key=cv2.contourArea)
    h, w = img.shape[:2]
    if len(max_contour) != 4 and cv2.isContourConvex(max_contour):
        return img  # non-quardrilateral contour, refuse to do anything
    if cv2.contourArea(max_contour) < h*w*0.25:
        return img  # contour too small, refuse to do anything
    # order the four-point contour
    pts = max_contour.reshape(4,2).astype(np.float32)
    clockwise_pts = np.zeros_like(pts)
    tlbr = np.sum(pts, axis=1)
    trbl = np.diff(pts, axis=1)
    clockwise_pts[0] = pts[np.argmin(tlbr)]  # top-left = smallest coord sum
    clockwise_pts[2] = pts[np.argmax(tlbr)]  # bottom-right = largest coord sum
    clockwise_pts[1] = pts[np.argmin(trbl)]  # top-right = most negative coord diff
    clockwise_pts[3] = pts[np.argmax(trbl)]  # bottom-left = most positive coord diff
    # measure the lengths of the quadrilateral for resize target
    len_top = np.linalg.norm(clockwise_pts[0] - clockwise_pts[1])
    len_bottom = np.linalg.norm(clockwise_pts[2] - clockwise_pts[3])
    len_left = np.linalg.norm(clockwise_pts[0] - clockwise_pts[3])
    len_right = np.linalg.norm(clockwise_pts[1] - clockwise_pts[2])
    target_w, target_h = int(max(len_top, len_bottom)-1), int(max(len_left, len_right)-1)
    # four-point perspective transform
    dst = np.array([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]], dtype=np.float32)
    print(clockwise_pts, dst)
    M = cv2.getPerspectiveTransform(clockwise_pts, dst)
    img = cv2.warpPerspective(img, M, (target_w, target_h))
    return img, "img = deskew_canny(img)"



