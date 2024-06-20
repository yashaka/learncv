import cv2


def RGB(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def RGBs(path):
    return cv2.split(RGB(path))


def gray(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
