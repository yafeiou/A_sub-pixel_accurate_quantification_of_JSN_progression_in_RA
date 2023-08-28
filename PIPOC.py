import cv2
import numpy as np
import scipy.fftpack
from scipy.optimize import leastsq

def PIPOC(img1, img2, segmentation1, segmentation2, areaNum):
    """
    Return the displacement for each region.

    Parameters
    ----------
    img1 & img2 : ndarray, single-channel image array.
    segmentation1 & segmentation2 : ndarray, single-channel segmentation matrix.
    areaNum : int
        The number of regions.

    Returns
    -------
    out : ndarray, shape([[displacement of x_axis, displacement of y_axis], reliability], areaNum)

    """
    returnArr = []

    # window function
    imgShape = img1.shape
    hy = np.hanning(imgShape[0])
    hx = np.hanning(imgShape[1])
    hw = hy.reshape(hy.shape[0], 1) * hx

    phase1 = scipy.fftpack.fft2(img1 * hw)
    phase1 = np.real(scipy.fftpack.ifft2(phase1 / np.abs(phase1)))
    phase2 = scipy.fftpack.fft2(img2 * hw)
    phase2 = np.real(scipy.fftpack.ifft2(phase2 / np.abs(phase2)))

    for i in range(areaNum):
        returnArr.append(poc(np.where(segmentation1 == i, phase1, 0), np.where(segmentation2 == i, phase2, 0)))
    return returnArr

def poc(img1, img2, mf = (7, 7)):
    center = tuple(np.array(img2.shape) / 2)
    m = np.floor(list(center))
    u = list(map(lambda x: x / 2.0, m))

    r = pocFun(img1, img2)

    # least-square fitting
    max_pos = np.argmax(r)
    peak = (np.int(max_pos / img1.shape[1]), max_pos % img1.shape[1])

    if np.abs(peak[0]-m[0]) > 20 or np.abs(peak[1]-m[1]) > 20:
        return [[0, 0], 0]

    fitting_area = r[peak[0] - mf[0]: peak[0] + mf[0] + 1,
                     peak[1] - mf[1]: peak[1] + mf[1] + 1]

    p0 = [0.5, -(peak[0] - m[0]) - 0.02, -(peak[1] - m[1]) - 0.02]
    y, x = np.mgrid[-mf[0]:mf[0] + 1, -mf[1]:mf[1] + 1]
    y = y + peak[0] - m[0]
    x = x + peak[1] - m[1]
    errorfunction = lambda p: np.ravel(pocfunc_model(p[0], p[1], p[2], r, u)(y, x) - fitting_area)
    plsq = leastsq(errorfunction, p0)
    return ([[plsq[0][2], plsq[0][1]], plsq[0][0]])

def pocFun(img1, img2):

    # compute 2d fft
    F = scipy.fftpack.fft2(img1)
    F = F /np.abs(F)
    G = scipy.fftpack.fft2(img2)
    G = G /np.abs(G)
    G_ = np.conj(G)
    R = F * G_

    return scipy.fftpack.fftshift(np.real(scipy.fftpack.ifft2(R)))


def pocfunc_model(alpha, delta1, delta2, r, u):
    N1, N2 = r.shape
    V1, V2 = list(map(lambda x: 2 * x + 1, u))
    return lambda n1, n2: alpha / (N1 * N2) * np.sin((n1 + delta1) * V1 / N1 * np.pi) * np.sin((n2 + delta2) * V2 / N2 * np.pi)\
                                            / (np.sin((n1 + delta1) * np.pi / N1) * np.sin((n2 + delta2) * np.pi / N2))

