import cv2 
import pickle
import numpy as np
from skimage.transform import resize


MODEL = pickle.load(open("model.p", "rb"))
EMPTY = True
NOT_EMPTY = False

def evaluate(spot):
    spot = resize(spot, (15, 15, 3))
    flat_data = np.array([spot.flatten()])

    y_predict = MODEL.predict(flat_data)
    if y_predict == 0:
        return EMPTY
    else:
        return NOT_EMPTY

def get_parking_spots_boxes(connected_components):
    total_labels, label_ids, values, centroid = connected_components

    slots = []
    coefficient = 1

    for i in range(1, total_labels):
        
        # extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coefficient)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coefficient)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coefficient)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coefficient)

        slots.append([x1, y1, w, h])

    return slots