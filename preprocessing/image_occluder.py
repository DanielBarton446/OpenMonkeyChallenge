import json
import cv2
import os
import numpy as np
import random


''' 
Example Code: 

def drawAllBlackBox(boxes: list, train_path):
    black_box = np.full((20, 20), 255)
    drawAllImages(boxes, black_box, train_path)


-----------------------------

    train_path = input(
        "Enter the path to training image folder/DIRECTORY\nDon't forget the '/' at the end of it\npath: ")

    # train_path = "train_images/"

    json_path = input(
        "Enter the path to the train annotations FILE\npath: ")
    # json_path = "./twenty_train_annotation.json"

    subsample_percentage = 0.2
    boxes = buildBoxImages(json_path, subsample_percentage)
    drawAllBlackBox(boxes, train_path)

'''


class TrainingImage():

    def __init__(self, file, landmarks: np.array, bbox: np.array):
        self.file = file
        self.landmarks = landmarks
        self.x = bbox[0]
        self.y = bbox[1]
        self.width = bbox[2]
        self.height = bbox[3]


def selectRandomKeypoint(training_img):
    return random.randint(0, len(training_img.landmarks)-1)


def buildBoxImages(json_filename, sample_percentage):
    f = open(json_filename)

    json_dump = json.load(f)
    data = json_dump["data"]

    if not os.path.exists("cands.npy"):
        print("Created new load file")
        entries = len(data)
        num_selections = int(entries * sample_percentage)
        candidates = np.random.choice(
            np.arange(entries), num_selections, replace=False)
        np.save("cands.npy", candidates)
    candidates = np.load("cands.npy")

    images = []
    for idx in candidates:
        cand = data[idx]
        filename = cand["file"]
        json_landmarks = cand["landmarks"]
        landmarks = []
        for idx in range(0, len(json_landmarks), 2):
            point = (json_landmarks[idx], json_landmarks[idx + 1])
            landmarks.append(point)
        bbox = np.array(cand["bbox"])
        images.append(TrainingImage(filename, landmarks, bbox))
    f.close()

    return images


def drawOcclusionCentered(train_img, occluder_img, dir_path):
    random_keypoint = selectRandomKeypoint(train_img)
    image = cv2.imread(dir_path + train_img.file)

    height, width = occluder_img.shape[0], occluder_img.shape[1]

    x, y = train_img.landmarks[random_keypoint]
    off_x, off_y = x-width//2, y-height//2

    for row in range(off_y, off_y + height - 1):
        for col in range(off_x, off_x + width - 1):
            if row < train_img.y + train_img.height \
                    and col < train_img.x + train_img.width:
                image[row][col] = occluder_img[row - off_y][col - off_x]

    img = image[train_img.y: train_img.y + train_img.height,
                train_img.x: train_img.x + train_img.width]
    output_dir = "./cropped_image/"
    if not os.path.exists(output_dir):
        print("MADE DIRECTORY: " + output_dir)
        os.makedirs(output_dir)

    cv2.imwrite(output_dir + train_img.file, img)
