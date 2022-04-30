import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import cv2
import image_occluder as imoc
import numpy as np


class BoxImage:
    def __init__(self, file_name, obj_type, xmin, xmax, ymin, ymax):
        self.file_name = file_name
        self.obj_type = obj_type
        self.xmin = int(float(xmin))
        self.xmax = int(float(xmax))
        self.ymin = int(float(ymin))
        self.ymax = int(float(ymax))

    def getSubImage(self):
        path = "../downloads/VOCdevkit/VOC2012/JPEGImages/"
        img = cv2.imread(path + self.file_name)
        img = img[self.ymin: self.ymax, self.xmin: self.xmax]
        return img

    def __str__(self):
        return str(self.xmin) + "\n" + \
            str(self.xmax) + "\n" + \
            str(self.ymin) + "\n" + \
            str(self.ymax) + "\n"


def makeImgBoxes(dir_path, obj_types):
    filenames = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    boxes = []
    for f in filenames:
        tree = ET.parse(dir_path + f)
        root = tree.getroot()
        image_file = root.find("filename").text
        for obj in root.findall('object'):
            obj_type = obj.find('name').text

            if (obj_type not in obj_types):  # ignore other types of images
                break

            box = obj.find('bndbox')
            my_box = BoxImage(image_file,
                              obj_type,
                              box.find('xmin').text,
                              box.find('xmax').text,
                              box.find('ymin').text,
                              box.find('ymax').text)
            boxes.append(my_box)

    return boxes

# TODO: scale the images nicer than this


def drawAllRegularOcclusions(train_images, occlusions, dir_path):
    occs = np.random.choice(np.arange(len(occlusions)), len(train_images))
    scale_percent = 0.2

    for idx, _ in enumerate(train_images):
        print(f"Image #: {idx} / {len(train_images)}\r", end="")
        occ_idx = occs[idx]
        sub_img = occlusions[occ_idx].getSubImage()
        train_img = train_images[idx]

        width = int(train_img.width * scale_percent)
        height = int(train_img.height * scale_percent)
        dim = (width, height)

        scaled_occluder = cv2.resize(
            sub_img, dim, interpolation=cv2.INTER_AREA)

        imoc.drawOcclusionCentered(train_img, scaled_occluder, dir_path)


def main():
    image_types = ["bird", "cat", "cow", "dog",
                   "person", "pottedplant", "sheep"]

    occluders = makeImgBoxes(
        "../downloads/VOCdevkit/VOC2012/Annotations/", image_types)

    # json_path = "./twenty_train_annotation.json"
    # json_path = input(
    # "Enter the path to the train annotations FILE\npath: ")
    json_path = "../downloads/Training/train_annotation.json"

    train_path = "../downloads/Training/train/"
    # train_path = input(
    # "Enter the path to training image folder/DIRECTORY\nDon't forget the '/' at the end of it\npath: ")

    sample_percentage = 0.2
    train_images = imoc.buildBoxImages(json_path, sample_percentage)

    drawAllRegularOcclusions(train_images, occluders, train_path)


if __name__ == '__main__':
    main()
