# coding=utf-8
import numpy as np
import cv2
import os

BORDER_PLACE_SIZE = 1120
BORDER_PADDING = 40
TEXT_PADDING = 250

def findFaces(imagePath, imageDebug=False):
    """
    find faces on photo given by input path and returns number of detected faces.
    :param imagePath: path to image
    :param imageDebug: if True, open image with detected faces in detached window
    :return:
    """
    face_cascade = cv2.CascadeClassifier(r'./cascades/haarcascade_frontalface_default.xml')

    img = cv2.imread(imagePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if imageDebug:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        res = cv2.resize(img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('detection', res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return len(faces)


def addBorder(imagePath, name, code, dstPath="./playerImages/", imageDebug=False):
    img = cv2.imread(imagePath)
    borderImage = cv2.imread(r"./imageSources/ramka.png")
    if imageDebug:
        cv2.imshow('border', cv2.resize(borderImage, None, fx=0.5, fy=0.5))
    # border2gray = cv2.cvtColor(borderImage, cv2.COLOR_BGRA2GRAY)
    # cv2.imshow('gray', cv2.resize(border2gray, None, fx=0.5, fy=0.5))
    # _, mask = cv2.threshold(border2gray, 100, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('mask', cv2.resize(mask, None, fx=0.5, fy=0.5))

    freePlace = borderImage[BORDER_PADDING:BORDER_PLACE_SIZE + BORDER_PADDING,
                BORDER_PADDING:BORDER_PLACE_SIZE + BORDER_PADDING]
    if imageDebug:
        cv2.imshow('inside', cv2.resize(freePlace, None, fx=0.5, fy=0.5))

    k = float(BORDER_PLACE_SIZE) / min(img.shape[:2])
    img = cv2.resize(img, None, fx=k, fy=k)

    if imageDebug:
        cv2.imshow('image before cut', cv2.resize(img, None, fx=0.5, fy=0.5))
    if img.shape[0] != img.shape[1]:
        borderCutSize = (max(img.shape[0], img.shape[1]) - BORDER_PLACE_SIZE) / 2
        img = img[borderCutSize:BORDER_PLACE_SIZE + borderCutSize] \
            if img.shape[0] > img.shape[1] \
            else img[:, borderCutSize:BORDER_PLACE_SIZE + borderCutSize]
    if imageDebug:
        cv2.imshow('image after cut', cv2.resize(img, None, fx=0.5, fy=0.5))
    borderImage[BORDER_PADDING:BORDER_PLACE_SIZE + BORDER_PADDING,
    BORDER_PADDING:BORDER_PLACE_SIZE + BORDER_PADDING] = img

    cv2.putText(borderImage, "ID: " + str(code), (TEXT_PADDING, 1400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)

    if imageDebug:
        cv2.imshow('result', cv2.resize(borderImage, None, fx=0.5, fy=0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(name, borderImage)
    if os.path.exists(dstPath + name):
        os.remove(dstPath + "/" + name)
    os.rename("./" + name, dstPath + "/" + name)


if __name__ == "__main__":
    print("Number of faces detected on test picture: {}".format(findFaces(r"./imageSources/test_image.jpg")))
addBorder(r"./imageSources/test_image.jpg", "test_photo_result.jpg", "1234-5678", imageDebug=True)
