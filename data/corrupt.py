# type: ignore
import argparse
# type: ignore
import h5py
import ctypes
import functools
import os
import warnings
from io import BytesIO
from typing import Any, List, Tuple

import cv2
import numpy as np  # type: ignore
import skimage as sk
import torch
import torchvision  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from PIL import Image as PILImage
from pkg_resources import resource_filename
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
from skimage.filters import gaussian
from wand.api import library as wandlibrary
from wand.image import Image as WandImage

# /////////////// Corruption Helpers ///////////////

warnings.simplefilter("ignore", UserWarning)


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (
    ctypes.c_void_p,  # wand
    ctypes.c_double,  # radius
    ctypes.c_double,  # sigma
    ctypes.c_double,
)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2 : mapsize : stepsize] = wibbledmean(
            ltsum
        )
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2 : mapsize : stepsize, 0:mapsize:stepsize] = wibbledmean(
            ttsum
        )

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))

    top = (h - ch) // 2
    img = scizoom(
        img[top : top + ch, top : top + ch], (zoom_factor, zoom_factor, 1), order=1
    )
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top : trim_top + h, trim_top : trim_top + h]


# /////////////// End Corruption Helpers ///////////////


# /////////////// Corruptions /////////////// ----- SEE END OF FILE FOR CORRUPT LOOPS

def gaussian_noise(x, severity=1, dim=28):
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.0
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1, dim=28):
    c = [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.0
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255


def impulse_noise(x, severity=1, dim=28):
    c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255.0, mode="s&p", amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1, dim=28):
    c = [0.15, 0.2, 0.35, 0.45, 0.6][severity - 1]

    x = np.array(x) / 255.0
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


# commented out because this was unused
# def fgsm(x, source_net, severity=1, dim=28):
#     c = [8, 16, 32, 64, 128][severity - 1]
#
#     x = V(x, requires_grad=True)
#     logits = source_net(x)
#     source_net.zero_grad()
#     loss = F.cross_entropy(
#         logits, V(logits.data.max(1)[1].squeeze_()), size_average=False
#     )
#     loss.backward()
#
#     return standardize(
#         torch.clamp(
#             unstandardize(x.data) + c / 255.0 * unstandardize(torch.sign(x.grad.data)),
#             0,
#             1,
#         )
#     )


def gaussian_blur(x, severity=1, dim=28):
    c = [1, 2, 3, 4, 6][severity - 1]

    x = gaussian(np.array(x) / 255.0, sigma=c, multichannel=True)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1, dim=28):
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255.0, sigma=c[0], multichannel=True) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(dim - c[1], c[1], -1):
            for w in range(dim - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255.0, sigma=c[0], multichannel=True), 0, 1) * 255


def defocus_blur(x, severity=1, dim=28):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = np.array(x) / 255.0
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    return np.clip(channels, 0, 1) * 255


def motion_blur(x, severity=1, dim=28):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

    output = BytesIO()
    x.save(output, format="PNG")
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)

    if x.shape != (dim, dim):
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def zoom_blur(x, severity=1, dim=28):
    c = [
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.02),
        np.arange(1, 1.26, 0.02),
        np.arange(1, 1.31, 0.03),
    ][severity - 1]

    x = (np.array(x) / 255.0).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, severity=1, dim=28):
    c = [(1.5, 2), (2.0, 2), (2.5, 1.7), (2.5, 1.5), (3.0, 1.4)][severity - 1]

    x = np.array(x) / 255.0
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:dim, :dim][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, severity=1, dim=28):
    c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
    idx = np.random.randint(5)
    filename = [
        resource_filename(__name__, "frost/frost1.png"),
        resource_filename(__name__, "frost/frost2.png"),
        resource_filename(__name__, "frost/frost3.png"),
        resource_filename(__name__, "frost/frost4.jpg"),
        resource_filename(__name__, "frost/frost5.jpg"),
        resource_filename(__name__, "frost/frost6.jpg"),
    ][idx]
    frost = cv2.imread(filename)
    # randomly crop and convert to rgb
    x_start, y_start = (
        np.random.randint(0, frost.shape[0] - dim),
        np.random.randint(0, frost.shape[1] - dim),
    )
    frost = frost[x_start : x_start + dim, y_start : y_start + dim][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)


def snow(x, severity=1, dim=28):
    c = [
        (0.1, 0.3, 3, 0.5, 10, 4, 0.8),
        (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
        (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
        (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
        (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55),
    ][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.0
    snow_layer = np.random.normal(
        size=x.shape[:2], loc=c[0], scale=c[1]
    )  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray(
        (np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode="L"
    )
    output = BytesIO()
    snow_layer.save(output, format="PNG")
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = (
        cv2.imdecode(
            np.fromstring(snow_layer.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED
        )
        / 255.0
    )
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(
        x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(dim, dim, 1) * 1.5 + 0.5
    )
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def spatter(x, severity=1, dim=28):
    c = [
        (0.65, 0.3, 4, 0.69, 0.6, 0),
        (0.65, 0.3, 3, 0.68, 0.6, 0),
        (0.65, 0.3, 2, 0.68, 0.5, 0),
        (0.65, 0.3, 1, 0.65, 1.5, 1),
        (0.67, 0.4, 1, 0.65, 1.5, 1),
    ][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.0

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1)) + 1e-6  # to avoid 0
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate(
            (
                175 / 255.0 * np.ones_like(m[..., :1]),
                238 / 255.0 * np.ones_like(m[..., :1]),
                238 / 255.0 * np.ones_like(m[..., :1]),
            ),
            axis=2,
        )

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0

        # mud brown
        color = np.concatenate(
            (
                63 / 255.0 * np.ones_like(x[..., :1]),
                42 / 255.0 * np.ones_like(x[..., :1]),
                20 / 255.0 * np.ones_like(x[..., :1]),
            ),
            axis=2,
        )

        color *= m[..., np.newaxis]
        x *= 1 - m[..., np.newaxis]

        return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1, dim=28):
    c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]

    x = np.array(x) / 255.0
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1, dim=28):
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]

    x = np.array(x) / 255.0
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def saturate(x, severity=1, dim=28):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = np.array(x) / 255.0
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1, dim=28):
    c = [25, 18, 15, 10, 7][severity - 1]

    output = BytesIO()
    x.save(output, "JPEG", quality=c)
    x = PILImage.open(output)

    return x


def pixelate(x, severity=1, dim=28):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    x = x.resize((int(dim * c), int(dim * c)), PILImage.BOX)
    x = x.resize((dim, dim), PILImage.BOX)

    return x


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, severity=1, dim=28):
    c = [
        (
            244 * 2,
            244 * 0.7,
            244 * 0.1,
        ),  # 244 should have been 224, but ultimately nothing is incorrect
        (244 * 2, 244 * 0.08, 244 * 0.2),
        (244 * 0.05, 244 * 0.01, 244 * 0.02),
        (244 * 0.07, 244 * 0.01, 244 * 0.02),
        (244 * 0.12, 244 * 0.01, 244 * 0.02),
    ][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.0
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ]
    )
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(
        image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101
    )

    dx = (
        gaussian(
            np.random.uniform(-1, 1, size=shape[:2]), c[1], mode="reflect", truncate=3
        )
        * c[0]
    ).astype(np.float32)
    dy = (
        gaussian(
            np.random.uniform(-1, 1, size=shape[:2]), c[1], mode="reflect", truncate=3
        )
        * c[0]
    ).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = (
        np.reshape(y + dy, (-1, 1)),
        np.reshape(x + dx, (-1, 1)),
        np.reshape(z, (-1, 1)),
    )
    return (
        np.clip(
            map_coordinates(image, indices, order=1, mode="reflect").reshape(shape),
            0,
            1,
        )
        * 255
    )

# /////////////// End Corruptions ///////////////


corruption_tuple = (
    gaussian_noise,
    shot_noise,
    impulse_noise,
    defocus_blur,
    glass_blur,
    motion_blur,
    zoom_blur,
    snow,
    frost,
    fog,
    brightness,
    contrast,
    elastic_transform,
    pixelate,
    jpeg_compression,
    speckle_noise,
    gaussian_blur,
    spatter,
    saturate,
)

corruption_dict = {corr_func.__name__: corr_func for corr_func in corruption_tuple}


def corrupt(x, severity=1, dim=28, corruption_name=None, corruption_number=-1):
    """
    :param x: image to corrupt; a 224x224x3 numpy array in [0, 255]
    :param severity: strength with which to corrupt x; an integer in [0, 5]
    :param corruption_name: specifies which corruption function to call;
    must be one of 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate';
                    the last four are validation functions
    :param corruption_number: the position of the corruption_name in the above list;
    an integer in [0, 18]; useful for easy looping; 15, 16, 17, 18 are validation corruption numbers
    :return: the image x corrupted by a corruption function at the given severity; same shape as input
    """
    if corruption_name:
        x_corrupted = corruption_dict[corruption_name](PILImage.fromarray(x), severity, dim=dim)
    elif corruption_number != -1:
        x_corrupted = corruption_tuple[corruption_number](PILImage.fromarray(x), severity, dim=dim)
    else:
        raise ValueError("Either corruption_name or corruption_number must be passed")

    return np.uint8(x_corrupted)


corruptions = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "pixelate",
    "saturate",
    "shot_noise",
    "spatter",
    "speckle_noise",
    "zoom_blur",
]


def get_random_corruptions(train_n: int, test_n: int) -> Tuple[List[str], ...]:
    if train_n + test_n != len(corruptions):
        raise ValueError(f"train_n and test_n need to equal corruptions len: {train_n + test_n}")

    idx = torch.randperm(len(corruptions))
    train_c = [v for (i, v) in enumerate(corruptions) if i in idx[:train_n].tolist()]
    test_c = [v for (i, v) in enumerate(corruptions) if i in idx[train_n:].tolist()]
    return train_c, test_c


def build_dataset(level: int, dim: int, name: str, images: np.array) -> np.array:
    """
        NOTE:
         - we need to run all
         - if you use ResNet, you should pad the image for the skip connections as laid out in
           appendix A.2 here (https://arxiv.org/pdf/1906.02530.pdf)
             image = tf.pad(image, [[4, 4], [4, 4], [0, 0]])
         - corruption levels are all 1-5 levels
    """

    # TODO:
    #  - training set should have random flips and crops added to it
    #     image = tf.image.random_flip_left_right(image)
    #     image = tf.image.random_crop(image, CIFAR_SHAPE)

    # NOTE: dhtd corruptions expect to be applied before float32 conversion.
    apply_corruption = functools.partial(corrupt, severity=level, dim=dim, corruption_name=name)
    images = np.stack([apply_corruption(im) for im in images])

    return images


def plot_corruptions(images: np.array, ds: str, corruption: str, level: int) -> None:
    fig, axes = plt.subplots(nrows=5, ncols=5)
    axes = axes.flatten()

    for (im, ax) in zip(images, axes):
        if len(images.shape) == 3:
            ax.imshow(im, cmap="binary")
        else:
            ax.imshow(im)

        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(f"{ds} corruption: {corruption} level: {level}")
    fig.savefig(os.path.join("charts", "corruptions", ds, f"{corruption}-{level}.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mnist-plot", action="store_true", help="generate the corrupted test set for MNIST")
    parser.add_argument("--cifar10-plot", action="store_true", help="generate the corrupted test set for CIFAR10")
    parser.add_argument("--omniglot-plot", action="store_true", help="generate the corrupted test set for Omniglot")
    parser.add_argument("--mnist", action="store_true", help="generate the corrupted test set for MNIST")
    parser.add_argument("--omniglot", action="store_true", help="generate the corrupted test set for Omniglot")
    parser.add_argument("--miniimagenet", action="store_true", help="generate the corrupted test set for MiniImageNet")
    parser.add_argument("--cifar10", action="store_true", help="generate the corrupted test set for CIFAR10")

    pargs = parser.parse_args()

    if pargs.mnist_plot:
        test = torchvision.datasets.MNIST("/home/jeff/datasets", download=True, train=False)
        test.name = "MNIST"

        d = test.data.numpy()
        d = np.expand_dims(d, 3)
        d = np.repeat(d, 3, 3)

        d = d[:25]
        for i in range(1, 6):
            for c in corruptions:
                print(f"MNIST {c}: {i}")
                images = build_dataset(i, 28, c, d)
                # eliminate the last channel which is needed for some of the corruptions
                plot_corruptions(images[:, :, :, 0], "MNIST", c, i)

    elif pargs.cifar10_plot:
        test = torchvision.datasets.CIFAR10("/home/jeff/datasets", download=True, train=False)
        test.name = "CIFAR10"

        d = test.data
        d = d[:25]
        for i in range(1, 6):
            for c in corruptions:
                print(f"CIFAR10 {c}: {i}")
                images = build_dataset(i, 32, c, d)
                plot_corruptions(images, "CIFAR10", c, i)

    elif pargs.omniglot_plot:
        raise NotImplementedError()

    elif pargs.miniimagenet:
        nat_dir = os.path.join("/home", "jeff", "datasets", "miniimagenet")

        dirs = os.listdir(nat_dir)
        for dset in ["val", "train", "test"]:
            dset_path = os.path.join(nat_dir, f"{dset}_data.hdf5")
            for c in corruptions:
                corr_path = os.path.join(nat_dir, f"{dset}_{c}")
                os.makedirs(corr_path, exist_ok=True)

                x = []
                data = h5py.File(dset_path)["datasets"]
                for img_cls in data:
                    x.append(data[img_cls])

                x = np.reshape(np.stack(x), (-1, 84, 84, 3))
                examples = []
                perm = np.random.permutation(12)[:6]
                for i in range(0, 6):
                    if i > 0:
                        x = build_dataset(i, 84, c, x)
                    examples.append(x[perm * 600])
                    print(f"corruption: {c} i: {i} saving x: {x.shape}")
                    np.save(os.path.join(corr_path, f"{i}.npy"), x)

                t = np.reshape(np.stack(examples), (-1, 84, 84, 3))
                t = torch.from_numpy(np.stack(t)).transpose(1, 3) / 255.0

                grid = torchvision.utils.make_grid(t, nrow=6)
                torchvision.utils.save_image(grid, os.path.join(corr_path, "example.png"))

    elif pargs.omniglot:
        nat_dir = os.path.join("/home", "jeff", "datasets", "omniglot-resized")
        c_dir = os.path.join("/home", "jeff", "datasets", "corruptions", "omniglot-resized")
        os.makedirs(c_dir, exist_ok=True)

        dirs = os.listdir(nat_dir)
        for dset in dirs:
            if ".zip" not in dset and ".py" not in dset:
                dset_path = os.path.join(nat_dir, dset)
                c_dset_path = os.path.join(c_dir, dset)
                os.makedirs(c_dset_path, exist_ok=True)
                for lang in os.listdir(dset_path):
                    lang_path = os.path.join(dset_path, lang)
                    c_lang_path = os.path.join(c_dset_path, lang)
                    os.makedirs(c_lang_path, exist_ok=True)
                    for char in os.listdir(lang_path):
                        char_path = os.path.join(lang_path, char)
                        c_char_path = os.path.join(c_lang_path, char)
                        os.makedirs(c_char_path, exist_ok=True)

                        imgs = os.listdir(char_path)
                        for c in corruptions:
                            corr_path = os.path.join(c_char_path, c)
                            os.makedirs(corr_path, exist_ok=True)

                            x: Any = []
                            for img in imgs:
                                x.append(np.array(PILImage.open(os.path.join(char_path, img), mode='r').convert('L')))

                            x = np.expand_dims(x, 3)
                            x = np.repeat(x, 3, 3)

                            for (img, name) in zip(x, imgs):
                                name = name.split(".")[0]
                                PILImage.fromarray(img[:, :, 0]).save(os.path.join(corr_path, f"{name}_0.png"))  # type: ignore

                            for i in range(1, 6):
                                print(f"{lang} {char} {c}: {i}")
                                images = build_dataset(i, 28, c, x)
                                for img_, name in zip(images, imgs):
                                    # eliminate the last channel which is needed for some of the corruptions
                                    name = name.split(".")[0]
                                    PILImage.fromarray(img_[:, :, 0]).save(os.path.join(corr_path, f"{name}_{i}.png"))  # type: ignore

    elif pargs.mnist:
        test = torchvision.datasets.MNIST("/home/jeff/datasets", download=True, train=False)
        test.name = "MNIST"

        x = test.data.numpy()
        x = np.expand_dims(x, 3)
        x = np.repeat(x, 3, 3)
        y = test.targets.numpy()

        savepath = os.path.join(os.sep, "home", "jeff", "datasets", "corruptions", "MNIST")
        for i in range(1, 6):
            for c in corruptions:
                print(f"MNIST {c}: {i}")
                images = build_dataset(i, 28, c, x)
                # eliminate the last channel which is needed for some of the corruptions
                np.save(os.path.join(savepath, f"x-{c}-{i}.npy"), images[:, :, :, 0])
                np.save(os.path.join(savepath, f"y-{c}-{i}.npy"), y)

    elif pargs.cifar10:
        test = torchvision.datasets.CIFAR10("/home/jeff/datasets", download=True, train=False)
        test.name = "CIFAR10"

        x = test.data
        y = np.array(test.targets)

        savepath = os.path.join(os.sep, "home", "jeff", "datasets", "corruptions", "CIFAR10")
        for i in range(1, 6):
            for c in corruptions:
                print(f"CIFAR10 {c}: {i}")
                images = build_dataset(i, 32, c, x)

                np.save(os.path.join(savepath, f"x-{c}-{i}.npy"), images)
                np.save(os.path.join(savepath, f"y-{c}-{i}.npy"), y)

    else:
        raise ValueError("something wierd")
