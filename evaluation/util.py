import numpy as np
import cv2
import os
from sys import stdout
import math
from scipy.spatial import KDTree
from shapely.geometry import Polygon


base_directory = "/Users/fmgarcia/Desktop/MapDetection/maps_project/detection_code/evaluation"

def retrieve_regions(filename):
    lines = open(filename, "r").readlines()
    regions = []
    regions_by_image = {}
    image_name = ""
    angle = 0.0
    for i, line in enumerate(lines):
        stdout.write("\rReading line %d/%d" %(i, len(lines)))
        stdout.flush()
        if line.endswith(".tiff\n"):
            if len(regions) > 0:
                regions_by_image[image_name] = list(regions)
            regions = []
            image_name = line.split("/")[-1][:-1]
        elif line.startswith("angle"):
            angle = float(line.split(" ")[-1][:-1])
        else:
            split_line = line.split(" ")
            if len(split_line) == 4:
                regions.append( (float(split_line[0]), float(split_line[1]), 
                    float(split_line[2]), float(split_line[3]), angle) )
            elif len(split_line) == 5:
                regions.append( (float(split_line[0]), float(split_line[1]), 
                    float(split_line[2]), float(split_line[3]), float(split_line[4]), angle) ) 

    regions_by_image[image_name] = list(regions)
    return regions_by_image


def vis_detections_bbox(image, predictions, threshold, annotations=None):
    for i in xrange(len(predictions)):
        bbox = predictions[i][:4]
        score = predictions[i][4]
        if score > threshold:
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0])+int(bbox[2]), int(bbox[1])-int(bbox[3])), (0, 255, 0), 5)
    
    if annotations is not None:
        for i in range(len(annotations)):
            bbox = annotations[i]
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0])+int(bbox[2]), int(bbox[1])-int(bbox[3])), (255, 0, 0), 2)

    return image


def vis_detections_pts(image, predictions, threshold, annotations=None):
    cnt1 = []
    for i in xrange(len(predictions)):
        score = predictions[i][-1]
        if score > threshold:
            pt1 = [int(predictions[i][0]), int(predictions[i][1])]
            pt2 = [int(predictions[i][2]), int(predictions[i][3])]
            pt3 = [int(predictions[i][4]), int(predictions[i][5])]
            pt4 = [int(predictions[i][6]), int(predictions[i][7])]

            cnt = np.array([pt1, pt2, pt3, pt4])
            cnt1.append(cnt)
            cv2.drawContours(image, [cnt], 0, (0, 0, 255), 5)

    if annotations is not None:
        for i in range(len(annotations)):
            pt1 = [int(annotations[i][0]), int(annotations[i][1])]
            pt2 = [int(annotations[i][2]), int(annotations[i][3])]
            pt3 = [int(annotations[i][4]), int(annotations[i][5])]
            pt4 = [int(annotations[i][6]), int(annotations[i][7])]

            ############# Visualize bounding boxes on original image ##################
            cnt = np.array([pt1, pt2, pt3, pt4])
            cv2.drawContours(image, [cnt], 0, (0, 0, 255), 2)

    return image, cnt1

def rotate_image(mat, angle, original_shape):
    height, width = original_shape[:2]
    image_center = (width // 2, height // 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, scale=1.0)

    # radians = math.radians(angle)
    # sin = math.sin(radians)
    # cos = math.cos(radians)
    # bound_w = int((height * abs(sin)) + (width * abs(cos)))
    # bound_h = int((height * abs(cos)) + (width * abs(sin)))
    #
    # rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    # rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])
    #
    # bounds = (bound_w, bound_h)
    bounds = (width, height)
    rotated_mat = cv2.warpAffine(mat, rotation_mat, bounds)
    return rotated_mat, rotation_mat, bounds


def adjust_image_size(map_img, padding_amount):
    translate = (0, 0)
    if map_img.shape[0] > map_img.shape[1]:
        half_height = (map_img.shape[0] - map_img.shape[1]) // 2
        padding = np.zeros( (map_img.shape[0], half_height, map_img.shape[2]) )
        map_img = np.hstack( (padding, map_img) )
        map_img = np.hstack( (map_img, padding ) )
        translate = (half_height+padding_amount, padding_amount)
    elif map_img.shape[1] > map_img.shape[0]:
        half_width = (map_img.shape[1] - map_img.shape[0]) // 2
        padding = np.zeros( (half_width, map_img.shape[1], map_img.shape[2]) )
        map_img = np.vstack((padding, map_img))
        map_img = np.vstack((map_img, padding ))
        translate = (padding_amount, half_width+padding_amount)

    vert_padding = np.zeros((map_img.shape[0], padding_amount, map_img.shape[2]))
    map_img = np.hstack((vert_padding, map_img))
    map_img = np.hstack((map_img, vert_padding))
    hor_padding = np.zeros((padding_amount, map_img.shape[1], map_img.shape[2]))
    map_img = np.vstack((hor_padding, map_img))
    map_img = np.vstack((map_img, hor_padding))
    return map_img, translate


def compute_intersection(rect_a, rect_b):
    A = [[rect_a[0],rect_a[1]], [rect_a[2],rect_a[3]], [rect_a[4],rect_a[5]], [rect_a[6],rect_a[7]]]
    B = [[rect_b[0],rect_b[1]], [rect_b[2], rect_b[3]], [rect_b[4], rect_b[5]], [rect_b[6], rect_b[7]]]

    output_polygon_shape = Polygon(A)
    input_polygon_shape = Polygon(B)
    intersection_area = input_polygon_shape.intersection(output_polygon_shape).area
    # print intersection_area
    return intersection_area

def compute_union(rect_a, rect_b, intersection):
    area_a = rect_a[2]*rect_a[3]
    area_b = rect_b[2]*rect_b[3]
    return area_a+area_b - intersection


def convert_bbox_format(bbox, angle, pivot):
    rad_angle = math.radians(angle)
    R = np.array([[math.cos(rad_angle), -math.sin(rad_angle)], [math.sin(rad_angle), math.cos(rad_angle)]])
    bl = R.dot(np.array([[bbox[0]-pivot[0]], [pivot[1]-bbox[1]] ]))
    br = R.dot(np.array([[(bbox[0]+bbox[2])-pivot[0]], [pivot[1]-bbox[1]] ]))
    tr = R.dot(np.array([[(bbox[0]+bbox[2])-pivot[0]], [pivot[1]-(bbox[1]-bbox[3])] ]))
    tl = R.dot(np.array([[bbox[0]-pivot[0]], [pivot[1]-(bbox[1]-bbox[3])]]))

    return (pivot[0] + bl[0,0], pivot[1] - bl[1,0],
            pivot[0] + br[0,0], pivot[1] - br[1,0],
            pivot[0] + tr[0,0], pivot[1] - tr[1,0],
            pivot[0] + tl[0,0], pivot[1] - tl[1,0])


def filter_predictions(predictions):
    predicted_points = [(x[0], x[1]) for x in predictions]
    indices = range(len(predictions))
    print(indices)
    filtered_points = predicted_points
    filtered_indices = indices
    done = False
    while done is False:
        print("Regions remaining" + str(len(filtered_points)))
        done = True
        kdtree = KDTree(filtered_points, leafsize=10)
        new_filtered_points = []
        new_filtered_indices = []
        used_points = []
        for k, pt in enumerate(filtered_points):
            if pt in used_points:
                continue

            matches = kdtree.query(pt, k=20, p=2)
            bbox_index = filtered_indices[k]
            candidate = pt
            candidate_bbox = predictions[bbox_index]
            width = math.sqrt( (candidate_bbox[2]-candidate_bbox[0])**2 + (candidate_bbox[3]-candidate_bbox[1])**2)
            height = math.sqrt( (candidate_bbox[6]-candidate_bbox[0])**2 + (candidate_bbox[7]-candidate_bbox[1])**2)

            candidate_area = width * height
            candidate_index = bbox_index
            candidate_ar = width / height if width > height else height / width
            print(matches[1])
            for i, match in enumerate(matches[1]):
                print(match)
                match_bbox_index = filtered_indices[match]
                match_bbox = predictions[match_bbox_index]
                width = math.sqrt((match_bbox[2] - match_bbox[0]) ** 2 + (match_bbox[3] - match_bbox[1]) ** 2)
                height = math.sqrt((match_bbox[6] - match_bbox[0]) ** 2 + (match_bbox[7] - match_bbox[1]) ** 2)

                match_area = width*height
                match_ar = width / height if width > height else height / width
                intersection = compute_intersection(match_bbox, candidate_bbox)

                if candidate_area >= match_area and intersection / match_area > 0.7:
                # if candidate_ar >= match_ar and intersection / candidate_area > 0.5:
                    used_points.append( filtered_points[match] )
                elif match_area > candidate_area and intersection / candidate_area > 0.7:
                # elif match_ar > candidate_ar and intersection / match_area > 0.5:
                    used_points.append(candidate)
                    candidate = filtered_points[match]
                    candidate_area = match_area
                    candidate_ar = match_ar
                    candidate_index = match_bbox_index
                    done = False

            new_filtered_points.append(candidate)
            new_filtered_indices.append(candidate_index)

        filtered_points = new_filtered_points
        filtered_indices = new_filtered_indices

    print("Total regions " + str(len(filtered_points)))
    return [predictions[k] for k in filtered_indices]


        
