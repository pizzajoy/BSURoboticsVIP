
import numpy as np
import cv2
from cameralnfo import CameraInfo
from lanes2 import LaneMarker
cap = cv2.VideoCapture(1)
croppedWidth = 320
croppedHeight = 240
camera_info = CameraInfo(53, 40, 76, 180, 217, croppedWidth, croppedHeight)
lanes_2 = LaneMarker(cap)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    output = lanes_2.process_image(frame)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   # output = camera_info.process_image(frame)
   # camera_info = CameraInfo(53, 40, 76, 180, 217, croppedWidth, croppedHeight)  # ground level# 3/4 out

    img_displayBirdsEye = camera_info.convertToFlat(output)

    # Display the resulting frame
    cv2.imshow('frame',img_displayBirdsEye)
   # cv2.imshow('frame', output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
