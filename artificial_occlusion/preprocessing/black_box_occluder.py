import image_occluder as imoc
import numpy as np


def drawAllBlackBox(train_images: list, train_path):
    scale_percent = 0.2

    for idx, _ in enumerate(train_images):
        print(f"Image #: {idx} / {len(train_images)}\r", end="")
        train_img = train_images[idx]

        width = int(train_img.width * scale_percent)
        height = int(train_img.height * scale_percent)
        dim = (width, height)

        scaled_occluder = np.full((height, width), 0)

        imoc.drawOcclusionCentered(train_img, scaled_occluder, train_path)


def main():
    # train_path = input(
    # "Enter the path to training image folder/DIRECTORY\nDon't forget the '/' at the end of it\npath: ")

    train_path = "../downloads/Training/train/"

    # train_path = "train_images/"

    # json_path = input(
    # "Enter the path to the train annotations FILE\npath: ")
    # json_path = "./twenty_train_annotation.json"

    json_path = "../downloads/Training/train_annotation.json"

    subsample_percentage = 0.2
    boxes = imoc.buildBoxImages(json_path, subsample_percentage)
    drawAllBlackBox(boxes, train_path)


if __name__ == "__main__":
    main()
