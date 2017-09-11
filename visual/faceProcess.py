# coding=utf-8
import numpy as np
import cv2

BORDER_PLACE_SIZE = 1120
BORDER_PADDING = 40


def findFaces(imagePath, imageDebug=False):
    """
    find faces on photo given by input path and returns number of detected faces.
    :param imagePath: path to image
    :param imageDebug: if True, open image with detected faces in detached window
    :return:
    """
    face_cascade = cv2.CascadeClassifier(r'.\cascades\haarcascade_frontalface_default.xml')

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


def addBorder(imagePath, imageDebug=False):
    img = cv2.imread(imagePath)
    borderImage = cv2.imread(r".\imageSources\ramka.jpg", cv2.IMREAD_UNCHANGED)
    cv2.imshow('border', cv2.resize(borderImage, None, fx=0.5, fy=0.5))

    border2gray = cv2.cvtColor(borderImage,cv2.COLOR_BGRA2GRAY)
    cv2.imshow('gray', cv2.resize(border2gray, None, fx=0.5, fy=0.5))
    _,mask = cv2.threshold(border2gray, 100, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('mask', cv2.resize(mask, None, fx=0.5, fy=0.5))
    freePlace = borderImage[BORDER_PADDING:BORDER_PLACE_SIZE + BORDER_PADDING,
                BORDER_PADDING:BORDER_PLACE_SIZE + BORDER_PADDING]
    cv2.imshow('inside', cv2.resize(freePlace, None, fx=0.5, fy=0.5))
    k = float(BORDER_PLACE_SIZE) / min(img.shape[:2])
    img = cv2.resize(img, None, fx=k, fy=k)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    borderImage[BORDER_PADDING:BORDER_PLACE_SIZE + BORDER_PADDING,
    BORDER_PADDING:BORDER_PLACE_SIZE + BORDER_PADDING] = img
    cv2.imshow('result', cv2.resize(borderImage, None, fx=0.5, fy=0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Number of faces detected on test picture: {}".format(findFaces(r".\imageSources\test_image.jpg")))
    addBorder(r".\imageSources\test_image.jpg", True)
