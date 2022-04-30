# Artificial Occlusion User Story

## Description

As a researcher,
I want there to be an option for using regular objects that will
occlude the image,
so that I can compare the effectiveness of black boxes versus regular objects.

## Acceptance Criteria

- selects a subset of images from the training set to apply artificial occlusion
  on
- artificial occlusion is performed by using the VOC dataset, 2012, available at: \
  &nbsp;[challenge](http://host.robots.ox.ac.uk/pascal/VOC/) \
  &nbsp;[dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) \
  &nbsp;[documentation](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.pdf)
- dataset images have standard geometric augmentation
  (scaling, rotation, translation, horizontal flip).
- the dataset images used are of the classes: \
  (sheep, potted plant, horse, dog, human, cow, cat, bird)

- the artificial occlusion covers at least one landmark from the training image

# Definition of Done

- passing unit testing
- passes testing per acceptance criteria items
- able to show feature in demo
