import numpy as np 
import os, sys
import math
import cv2

sys.path.append("../../evaluation/")
from util import rotate_image, adjust_image_size
import pickle

curr_path = os.path.dirname(os.path.realpath(__file__))

annotations_dir = sys.argv[1]
map_input_dir = sys.argv[2]
map_output_dir = os.path.join(curr_path, 'img')

files = os.listdir(annotations_dir)

fold = 5
total_maps = 0

maps = {}
for f in files:
    if f.endswith(".npy"):
        total_maps += 1
        data = np.load(os.path.join(annotations_dir, f)).item()
        map_name = f.split(".")[0]
        maps[map_name] = []
        for key in data.keys():
            vertices = data[key]['vertices']
            if len(vertices) == 4:
                maps[map_name].append(vertices)

#angles = range(-100, 105, 5)
angles = [0]

print("Generating angles...")
angles_by_map = {}
for map_name in maps.keys():
    vertices = maps[map_name]
    vertices_by_angle = {}
    for angle in angles:
        vertices_by_angle[angle] = list()

    for box in vertices:
        bottom_left = box[0]
        bottom_right = box[1]
        #### Center bottom left to origin
        y = bottom_right[1] - bottom_left[1]
        x = bottom_right[0] - bottom_left[0]
        angle = np.arctan2(y, x) * 180.0 / np.pi

        for a in vertices_by_angle.keys():
            if math.fabs(a-angle) <= 10:
                vertices_by_angle[a].append( (box, angle) )

    angles_by_map[map_name] = vertices_by_angle



if os.path.isdir(map_output_dir) == False:
    os.mkdir(map_output_dir)

current_fold = 0
annotations = None
for k, mapname in enumerate(angles_by_map.keys()):
    '''
    print fold
    print total_maps
    print total_maps/fold
    '''
    if k % (total_maps // fold) == 0:
        current_fold += 1
        fold_dir = "./fold_"+str(current_fold)
        if os.path.isdir(fold_dir) == False:
            os.mkdir(fold_dir)

        if annotations is not None:
            annotations.close()
        annotations = open(fold_dir+"/test.txt", "w")

    #print("Writing map " + mapname)
    map_img = cv2.imread(os.path.join(map_input_dir, mapname + ".tiff"))
    ######## Make room to rotate the image ####################
    padding_amount = 0
    #map_img, translate = adjust_image_size(map_img, padding_amount)
    #######################################################
    translate = (0, 0)
    original_shape = map_img.shape

    #print("Translating annotations")
    for angle in angles:
        for i in range(len(angles_by_map[mapname][angle])):
            bbox, bbox_angle = angles_by_map[mapname][angle][i]
            angles_by_map[mapname][angle][i][0][0] = (int(bbox[0][0]) + translate[0], int(bbox[0][1]) + translate[1])
            angles_by_map[mapname][angle][i][0][1] = (int(bbox[1][0]) + translate[0], int(bbox[1][1]) + translate[1])
            angles_by_map[mapname][angle][i][0][2] = (int(bbox[2][0]) + translate[0], int(bbox[2][1]) + translate[1])
            angles_by_map[mapname][angle][i][0][3] = (int(bbox[3][0]) + translate[0], int(bbox[3][1]) + translate[1])

    for angle in angles:
        #print("Annotating angle " + str(angle))
        bboxes = angles_by_map[mapname][angle]
	
        rot_image_name = mapname + "_" + str(angle) + ".tiff"

        if len(bboxes) > 0:
            annotations.write(os.path.join(map_output_dir, rot_image_name) + "\n")
            annotations.write(str(len(bboxes))+"\n")

            rot_img, rot_mat, bounds = rotate_image(map_img, angle, original_shape)
            for bbox, bbox_angle in bboxes:
                pt1 = [int(bbox[0][0]), int(bbox[0][1])]
                pt2 = [int(bbox[1][0]), int(bbox[1][1])]
                pt3 = [int(bbox[2][0]), int(bbox[2][1])]
                pt4 = [int(bbox[3][0]), int(bbox[3][1])]

                ############# Visualize bounding boxes on original image ##################
                # cnt = np.array([pt1, pt2, pt3, pt4])
                # cv2.drawContours(map_img, [cnt], 0, (255, 0, 0), 5)
                ##########################################################################



                #### Bounding boxes are given as coordinates on the original horizontal image ######
                #### I need to rotate the image, just as the network will see it as input, and determine the new coordinates of bounding boxes ####
                points = np.array([np.asarray([pt1[0], pt1[1], 1.0]),
                            np.asarray([pt2[0], pt2[1], 1.0]),
                            np.asarray([pt3[0], pt3[1], 1.0]),
                            np.asarray([pt4[0], pt4[1], 1.0])
                            ])

                transformed_points = rot_mat.dot(points.T).T


                ########## Visualize mapped bounding boxes after rotation ###############
                pt1 = [int(transformed_points[0][0]), int(transformed_points[0][1])]
                pt2 = [int(transformed_points[1][0]), int(transformed_points[1][1])]
                pt3 = [int(transformed_points[2][0]), int(transformed_points[2][1])]
                pt4 = [int(transformed_points[3][0]), int(transformed_points[3][1])]
                #
                # uncomment the following two lines to visualize the bounding boxes 
                # in the new set of images
                #cnt = np.array([pt1, pt2, pt3, pt4])
                #cv2.drawContours(rot_img, [cnt], 0, (255, 0, 0), 7)
                #########################################################################

                width =  math.sqrt( (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 )
                height = math.sqrt( (pt1[0] - pt4[0])**2 + (pt1[1] - pt4[1])**2 )
                string = str(pt1[0]) + " " + str(pt1[1]) + " " + str(width) + " " + str(height)
                annotations.write(string+"\n")

            cv2.imwrite(os.path.join(map_output_dir, rot_image_name), rot_img)

#        if os.path.isdir("./"+mapname) == False:
 #           os.mkdir("./"+mapname)

  #      cv2.imwrite("./"+mapname+"/"+str(angle)+".png", map_img)
