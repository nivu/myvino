import cv2 as cv
from math import cos, sin, pi
import numpy as np

winName = 'OpenVINO on Raspberry Pi'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
faceDetectionNet = cv.dnn.readNet('models/face-detection-retail-0004.xml',
                                  'models/face-detection-retail-0004.bin')
identityNet = cv.dnn.readNet('models/face-reidentification-retail-0095.xml',
                             'models/face-reidentification-retail-0095.bin')
faceDetectionNet.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
identityNet.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
cap = cv.VideoCapture(0)

while cv.waitKey(1) != 27:
    hasFrame, frame = cap.read()
    if not hasFrame:
        break
    frameHeight, frameWidth = frame.shape[0], frame.shape[1]
    # Detect faces on the image.
    blob = cv.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv.CV_8U)
    faceDetectionNet.setInput(blob)
    detections = faceDetectionNet.forward()
    for detection in detections.reshape(-1, 7):
        confidence = float(detection[2])
        if confidence > 0.5:
            xmin = int(detection[3] * frameWidth)
            ymin = int(detection[4] * frameHeight)
            xmax = int(detection[5] * frameWidth)
            ymax = int(detection[6] * frameHeight)
            xmax = max(1, min(xmax, frameWidth - 1))
            ymax = max(1, min(ymax, frameHeight - 1))
            xmin = max(0, min(xmin, xmax - 1))
            ymin = max(0, min(ymin, ymax - 1))
            # Run head pose estimation network.
            face = frame[ymin:ymax+1, xmin:xmax+1]
            blob = cv.dnn.blobFromImage(face, size=(60, 60), ddepth=cv.CV_8U)
            identityNet.setInput(blob)
            identityOp = identityNet.forward()
            print(identityOp)

    cv.imshow(winName, frame)
