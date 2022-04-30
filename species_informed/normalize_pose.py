import numpy as np
def get_row(obj):
    landmarks = obj['landmarks']
    bbox = obj['bbox']
    x, y, w, h = bbox
    
#     extract hip landmark
    hip = landmarks[11 * 2: 11 * 2 + 2]
    x_hip, y_hip = hip
    
#     get x and y coordinates of each landmark
    xs = np.array([landmarks[i] for i in range(0, 17 * 2, 2)])
    ys = np.array([landmarks[i] for i in range(1, 17 * 2, 2)])
    
#     use hip as origin of landmarks
    xs -= x_hip
    ys -= y_hip
    
#     scale relative landmarks with bbox
    xs = xs / w
    ys = ys / h
    
#     rotate points such that the head is at 0 degrees
    head_x, head_y = xs[3], ys[3]
    theta = -1 * np.arctan2(head_y, head_x)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    xy_mat = np.stack([xs, ys])
    rotated = rot_mat @ xy_mat
    return rotated.flatten()
