
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as mat
import matplotlib.pyplot as plt
import caffe
import cv2
import json
import math
# MODEL = 'ILSVRC' # ImageNet, don't use ImageNet, it wasn't trained on all categories
MODEL = 'coco'  # MS-Coco
MODEL = 'ours'
IMAGE_SIZE = 300  # 300x300 trained on coco or ILSVRC
# I wonder if we can take the coco model and further train it on
# http://image-net.org/synset?wnid=n02773838
# IMAGE_SIZE = 512 # for 512x512 trained on coco
# for detection - percentage that the model is sure it's what you're looking for
THRESHOLD = 0.20
# There are 21 categories.... pick one color for each
# just a tool for label finding


# for checking if a list contains elements of another
def any_in(a, b): return bool(set(a).intersection(b))


# for picking colors of the boxes
COLORS = plt.cm.hsv(np.linspace(0, 1, 255)).tolist()
caffe.set_device(0)
caffe.set_mode_gpu()
# caffe.set_mode_cpu()

get_ipython().magic(u'matplotlib inline')


# In[2]:


from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load COCO labels
if MODEL == 'ours':
    labelmap_file = 'data/coco/labelmap_coco2.prototxt'
if MODEL == 'coco':
    labelmap_file = 'data/coco/labelmap_coco.prototxt'
else:
    labelmap_file = 'data/ILSVRC2016/labelmap_ilsvrc_det.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames


# In[3]:


def loadmodel():
    if IMAGE_SIZE == 300 and MODEL == 'coco':
        model = 'deploy300.prototxt'
        weights = 'VGG_coco_SSD_300x300_iter_400000.caffemodel'
    elif IMAGE_SIZE == 512 and MODEL == 'coco':
        model = 'deploy512.prototxt'
        weights = 'VGG_coco_SSD_512x512_iter_360000.caffemodel'
    else:
        model = 'deploy2017.prototxt'
        weights = 'VGG_coco_SSD_300x300_iter_60000.caffemodel'
    # how you load a model with weights in Caffe
    return caffe.Net(model, weights, caffe.TEST)


# In[4]:


def preprocess(frame):
    # Frame must be IMG_SIZExIMG_SIZEx3
    frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE),
                       interpolation=cv2.INTER_LANCZOS4)
    # Frame must then be 3xHxW
    if len(frame.shape) == 3:
        frame = frame.transpose((2, 0, 1))
    return frame


# In[5]:


def detect(image, net):    # (Batch size, channels, Image size, Image size)
    # I wonder if we can increase the batch size and
    # put a list of images together, but I guess that's more for training
    net.blobs['data'].reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    # Transform the image to 1x3xSxS
    net.blobs['data'].data[0, ...] = image
    # See ssd_detect.ipynb from Wei Liu, author of SSD
    # https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_detect.py
    detections = net.forward()['detection_out']
    # Parse the output tensors
    det_label = detections[0, 0, :, 1]

    det_conf = detections[0, 0, :, 2]  # confidence
    det_xmin = detections[0, 0, :, 3]  # for bounding boxes per frame
    det_ymin = detections[0, 0, :, 4]
    det_xmax = detections[0, 0, :, 5]
    det_ymax = detections[0, 0, :, 6]

    # Instead of choosing a threshold, we keep all detections
    top_indices = [i for i, conf in enumerate(det_conf)]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)

    return (det_xmin, det_ymin, det_xmax, det_ymax, det_conf, top_labels, top_label_indices)


# In[6]:

def calcDist(coords1, coords2):
    return np.linalg.norm(coords1 - coords2)


# In[7]:


def loadvideo(filename, net):
    cap = cv2.VideoCapture(filename)
    # Actually doesn't store real frames, but the frame shape of midpoint vectors
    saved_frames = []
    FUZZY_MATCH = 5
    FRAMES_TO_HOLD = 15
    OWNER_DISTANCE = 50

    while cap.isOpened():
        ret, frame = cap.read()
        if np.any(frame != 0):
            bag_updated = []
            person_updated = []
            frame_processed = preprocess(frame)
            processed_det = detect(frame_processed, net)
            top_xmin, top_ymin, top_xmax, top_ymax, top_conf, top_labels, top_label_indices = processed_det
            # Midpoint_boxes is a tensor, which has the area of the frame from the video
            # But the value at each pixels position is only valid when it represents the midpoint of a detected box
            # The values will be width, height, label, and "owner_y, owner_x" which is set to the coordinates
            # of the person who is first within the threshold of what we consider owner if label is a bag
            midpoint_boxes = np.empty((frame.shape[0], frame.shape[1], 5))
            midpoint_boxes.fill(np.nan)
            for i in range(top_conf.shape[0]):
                xmin = int(round(top_xmin[i] * frame.shape[1]))
                ymin = int(round(top_ymin[i] * frame.shape[0]))
                xmax = int(round(top_xmax[i] * frame.shape[1]))
                ymax = int(round(top_ymax[i] * frame.shape[0]))
                score = top_conf[i]
                label = int(top_label_indices[i])
                label_name = top_labels[i]
                display_txt = '%s: %.2f' % (label_name, score)
                width = xmax - xmin + 1
                height = ymax - ymin + 1
                midx = xmin + (width) / 2
                midy = ymin + (height) / 2
                if label in [1, 27, 31, 33] and score > 0.1:
                    obj_array = np.asarray(
                        [width, height, label, np.nan, np.nan])
                    midpoint_boxes[midy, midx] = obj_array
                    found = -1
                    if len(saved_frames) >= 1:
                        for j in range(len(saved_frames) - 1, -1, -1):
                            fuzzy_min = midy - FUZZY_MATCH if midy >= FUZZY_MATCH else 0
                            fuzzx_min = midx - FUZZY_MATCH if midx >= FUZZY_MATCH else 0
                            fuzzy_max = midy + FUZZY_MATCH if midy + \
                                FUZZY_MATCH < frame.shape[1] else frame.shape[1] - 1
                            fuzzx_max = midx + FUZZY_MATCH if midx + \
                                FUZZY_MATCH < frame.shape[0] else frame.shape[0] - 1

                            sub_sample = saved_frames[j][fuzzy_min:fuzzy_max,
                                                         fuzzx_min:fuzzx_max]
                            for row in range(sub_sample.shape[0]):
                                box = sub_sample[row]
                                if np.isfinite(box.flatten()).any():
                                    for col in range(box.shape[0]):
                                        # Previous some-odd frame at position [j][row, col]
                                        pixel_midpoint = box[col]
                                        # If person or object is ocluded match width _OR_ height being similar
                                        if obj_array[0] - FUZZY_MATCH < pixel_midpoint[0] < obj_array[0] + FUZZY_MATCH or obj_array[1] - FUZZY_MATCH < pixel_midpoint[1] < obj_array[1] + FUZZY_MATCH and obj_array[2] == pixel_midpoint[2]:
                                            midpoint_boxes[midy,
                                                           midx][3:5] = pixel_midpoint[3:5]
                                            if label == 1:
                                                item_type = 1
                                                person_updated.append(
                                                    (row, col, midy, midx, pixel_midpoint[3], pixel_midpoint[4]))
                                            # prev location, new loc
                                            else:
                                                bag_updated.append(
                                                    (row, col, midy, midx, pixel_midpoint[3], pixel_midpoint[4]))
                                            # For person/bag row/col means the place the object previously was
                                            found = (j, row, col, midy,
                                                     midx, label, pixel_midpoint[3], pixel_midpoint[3])
                                            break
                                if found != -1:
                                    break
                            if found != -1:
                                break
                        # Currently have in found the layer where the bag or person was last seen
                        # found a person, check if person has moved and see if bag has also been identified
                        # found a person, see if the bag was already found in this frame
                        if found != -1 and found[5] == 1:
                            for bag in bag_updated:
                                if bag[0] == found[6] and bag[1] == found[7]:
                                    # Bag's old owner position was this old owner's position
                                    midpoint_boxes[bag[2],
                                                   bag[3], 3:5] = found[3:5]  # Now new owner's position is held by bag
                        elif found != -1:  # must be a bag that we found in this frame, see if the owner was updated
                            for person in person_updated:
                                if person[0] == found[6] and person[1] == found[7]:
                                    # Bag's old owner position was this old owner's position
                                    midpoint_boxes[person[2],
                                                   person[3], 3:5] = found[3:5]  # Now new owner's position is held by bag
                        if found != -1 and found[0] < FRAMES_TO_HOLD - 2:
                            # Must have skipped a frame so add in relevant middle position
                            missing_frames = FRAMES_TO_HOLD - found[0]
                            diff_rows = found[3] - found[1]
                            diff_cols = found[4] - found[2]
                            # May be -b, or 0 , or +a
                            incr_rows_per_frame = diff_rows // missing_frames
                            incr_cols_per_frame = diff_cols // missing_frames
                            makeup_mul = 0
                            for makeup_i in range(found[0], FRAMES_TO_HOLD):
                                makeup_mul += 1
                                saved_frames[makeup_i][found[1] + incr_rows_per_frame * makeup_mul, found[2] +
                                                       incr_cols_per_frame * makeup_mul] = saved_frames[found[0]][found[1], found[2]]
                        if found == -1:
                            # First time seeing the object, add
                            if label == 1:  # First time seeing person
                                person_updated.append(
                                    (np.nan, np.nan, midy, midx, np.nan, np.nan))
                            else:
                                bag_updated.append(
                                    (np.nan, np.nan, midy, midx, np.nan, np.nan))
            if len(saved_frames) == 0:
                # Do initial attribution of owners
                for i in range(frame.shape[0]):
                    for j in range(frame.shape[1]):
                        if midpoint_boxes[i, j, 0] != np.nan and midpoint_boxes[i, j, 2] in [27, 31, 33]:
                            min_i = i - \
                                OWNER_DISTANCE if (
                                    i - OWNER_DISTANCE) > 0 else 0
                            max_i = i + \
                                OWNER_DISTANCE if (
                                    i + OWNER_DISTANCE) < frame.shape[0] else frame.shape[0]
                            min_j = j - \
                                OWNER_DISTANCE if (
                                    j - OWNER_DISTANCE) > 0 else 0
                            max_j = j + \
                                OWNER_DISTANCE if (
                                    j + OWNER_DISTANCE) < frame.shape[1] else frame.shape[1] - 1
                            found_owner = false
                            potential_owners = []
                            bag_coord = np.asarray([i, j])
                            for y in range(min_i, max_i):
                                for x in range(min_j, max_j):
                                    if midpoint_boxes[y, x, 0] != np.nan and midpoint_boxes[y, x, 2] == 1:
                                        # y,x may be owner
                                        potential_owners.append(
                                            (y, x, calcDist(np.asarray([y, x]), bag_coord)))
                            potential_owners = sorted(
                                potential_owners, cmp=lambda a, b: a[2] - b[2])
                            print(potential_owners)
            for bag in bag_updated:
                if bag[0] == np.nan:
                    # new bag not seen before
                    bag_coord = np.asarray([bag[2], bag[3]])
                    min_i = bag[2] - \
                        OWNER_DISTANCE if (
                            bag[2] - OWNER_DISTANCE) > 0 else 0
                    max_i = bag[2] + \
                        OWNER_DISTANCE if (
                            bag[2] + OWNER_DISTANCE) < frame.shape[0] else frame.shape[0]
                    min_j = bag[3] - \
                        OWNER_DISTANCE if (
                            bag[3] - OWNER_DISTANCE) > 0 else 0
                    max_j = bag[3] + \
                        OWNER_DISTANCE if (
                            bag[3] + OWNER_DISTANCE) < frame.shape[1] else frame.shape[1] - 1
                    potential_owners = []
                    for i in range(min_i, max_i):
                        for j in range(min_j, max_j):
                            # look for person
                            if midpoint_boxes[i, j, 2] == 1:
                                # not a nan item, and is a person
                                potential_owners.append(i, j, calcDist(
                                    np.asarray([i, j]), bag_coord))
                    potential_owners = sorted(
                        potential_owners, cmp=lambda a, b: a[2] - b[2])
                    print(potential_owners)
            saved_frames.append(midpoint_boxes)
            if len(saved_frames) > FRAMES_TO_HOLD:
                saved_frames = saved_frames[1:]
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print 'how did we break?'
            break


# In[8]:


net = loadmodel()
loadvideo('AVSS_AB_Easy_Clipped.mov', net)
print('Finished!')


# In[80]:


# no longer outputs the images here, but they are all in the directory
