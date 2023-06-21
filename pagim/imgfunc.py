"""Image processing function"""

import os
import functools

import cv2
import numpy as np
import super_image
import torch

from .dewrap import dewrap


if torch.cuda.is_available():
    device = torch.device("cuda:0")
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


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


@functools.lru_cache
def get_edsr_base():
    """Singleton to load EDSR-Base model from Huggingface"""
    return super_image.EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4).to(device)



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


@image_op("Hough line transform from Canny edge")
def hough_edge(img):
    # blur to remove small noise first, then Canny for edges
    blur = cv2.GaussianBlur(img, (5,5), 0)
    canny = cv2.Canny(blur, 75, 200)
    # Probabilistic Hough line, with resolution at rho=1 theta=1 deg, threshold
    # for 50 votes, min line length 50px and max line gap 10px
    lines = cv2.HoughLinesP(canny, 1, np.pi/180, 50, None, 50, 10)
    if lines is None:
        return img, ""
    for line in lines:
        x0, y0, x1, y1 = line[0]
        print(line, np.arctan2(y1-y0, x1-x0)*180/np.pi)
        cv2.line(img, (x0,y0), (x1,y1), (255,0,0), 3, cv2.LINE_AA)
    return img, "img = hough_edge(img)"


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


# Improvement

@image_op("Gamma adjust")
def gamma_adj(img):
    gamma = 1.1
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    code = 'img = cv2.LUT(img, np.uint8([((i/255.0) ** (1/1.1)) * 255 for i in range(256)]))'
    img = cv2.LUT(img, table)
    return img, code


@image_op("Unsharp mask")
def usm(img):
    sigma = 2.0    # blur kernel stdev to calculate the kernel
    weight = -1.0  # USM needs negative weight
    blur = cv2.GaussianBlur(img, (0,0), sigma)
    img = cv2.addWeighted(img, 1-weight, blur, weight, 0).astype(np.uint8)
    code = "img = cv2.addWeighted(img, 2.0, cv2.GaussianBlur(img, (0,0), 2.0), -1.0, 0).astype(np.uint8)"
    return img, code


@image_op("Shadow removal")
def shadow_removal(img):
    channels = cv2.split(img)
    out_channels = []
    for c in channels:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        bg = cv2.medianBlur(cv2.dilate(c, kernel), 21)
        diff = 255 - cv2.absdiff(c, bg)
        # optional
        c = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        out_channels.append(c)
    img = cv2.merge(out_channels)
    return img, "img = shadow_removal(img)"


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
    blur = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (3,3), 0)
    blur = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    blur = cv2.fastNlMeansDenoising(blur, 11, 31, 9)
    # optional: assume page is light compared to the background, perform Otsu
    # threshold to binarize at high threshold 250 to make most background dark
    # so Canny edge detection can be more focused on the page
    #   _, thresh = cv2.threshold(blur, 250, 255, cv2.THRESH_OTSU)
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
        return img, ""  # non-quardrilateral contour, refuse to do anything
    if cv2.contourArea(max_contour) < h*w*0.25:
        return img, ""  # contour too small, refuse to do anything
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
    M = cv2.getPerspectiveTransform(clockwise_pts, dst)
    img = cv2.warpPerspective(img, M, (target_w, target_h))
    return img, "img = deskew_canny(img)"


@image_op("Deskew page using Hough line")
def deskew_hough(img):
    """Complete workflor from RGB image to deskewed"""
    # Blur and read Canny edge
    blur = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (3,3), 0)
    edged = cv2.Canny(blur, 50, 150, apertureSize=7)
    # Read Hough lines


@image_op("Derotate page using Otsu thresholding")
def derotate(img):
    """Complete workflow from RGB image to orthogonalized"""
    # Gaussian blur, threshold, and dilate the image
    blur = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (3,3), 0)
    _, blur = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INF + cv2.THRESH_OTSU)  # using Otsu thresholding
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    blur = cv2.dilate(blur, kernel, iterations=5)
    # Find the contour from the thresholded image
    contours, _hierarchy = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    # Find the min bounding rectangle, and from it, the angle
    rect = cv2.minAreaRect(max_contour)
    # to get the 4 corners: corners = cv2.boxPoints(rect)
    # rotate
    angle = rect[-1]
    angle = (90+angle) if angle < -45 else (90-angle) if angle > 45 else angle
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img, "img = derotate(img)"


@image_op("Downsample 2x")
def downsample(img):
    return cv2.pyrDown(img), "img = cv2.pyrDown(img)"


@image_op("EDSR-Base super-res")
def edsr_base(img):
    # make RGB image with pixel range 0-1 of float32 and in NCHW tensor
    inputs = torch.as_tensor([(np.float32(img) / 255.0).transpose([2,0,1])])
    # run the model over PyTorch
    model = get_edsr_base()
    preds = model(inputs.to(device))
    # transform PyTorch tensor back to numpy in HWC format
    pred = np.clip(1-preds.data.cpu().numpy()[0].transpose([1,2,0]), 0, 1) * 255.0
    img = np.int8(pred)
    return img, "img = EDSRBase(img)"


@image_op("Cubic dewrap")
def cubic_dewrap(img):
    return dewrap(img), "img = cubic_dewrap(img)"
