import cv2
# from PIL import Image
from utils import get_parking_spots_boxes, evaluate
import numpy as np

# calculate how similar images are
def calc_diffs(img1, img2):
    return np.abs(np.mean(img1) - np.mean(img2))

path = "/Users/gokcenazakyol/Desktop/parking-spot-detection/data/parking_1920_1080_loop.mp4"
crop_mask_path = "/Users/gokcenazakyol/Desktop/parking-spot-detection/data/mask.png"
"""
# resize mask
crop_mask = Image.open(crop_mask_path)
crop_mask = crop_mask.resize((1920, 1080))
crop_mask.save(crop_mask_path)
"""
crop_mask = cv2.imread(crop_mask_path, 0)

# apply thresholding to convert grayscale to binary image
ret,crop_mask = cv2.threshold(crop_mask,200,255,200)


cap = cv2.VideoCapture(path)
connected_components = cv2.connectedComponentsWithStats(image=crop_mask, connectivity=4, ltype=cv2.CV_32S)

spots = get_parking_spots_boxes(connected_components)
spots_status = [None for j in spots]
diffs = [None for j in spots]

previous_frame = None

frame_nmr = 0
ret = True
step = 30
while ret:
    ret, frame = cap.read()

    # changing status 
    if frame_nmr % step == 0:
        for spot_index, spot in enumerate(spots):
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            spot_status = evaluate(spot_crop)
            spots_status[spot_index] = spot_status

    # drawing rectangles
    for spot_index, spot in enumerate(spots):
        spot_status = spots_status[spot_index]
        x1, y1, w, h = spots[spot_index]
        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()
