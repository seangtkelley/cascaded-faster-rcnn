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
map_output_dir = os.path.join(os.sep, 'mnt', 'nfs', 'work1', 'elm', 'sgkelley', 'data', 'maps', 'angles_-90to90step5')

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
            maps[map_name].append(vertices)

angles = range(-90, 95, 5)
#angles = [0]

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

    print("Writing map " + mapname)
    map_img = cv2.imread(os.path.join(map_input_dir, mapname + ".tiff"))
    ######## Make room to rotate the image ####################
    padding_amount = 0
    map_img, translate = adjust_image_size(map_img, padding_amount)
    #######################################################
    #translate = (0, 0)
    original_shape = map_img.shape

    print("Translating annotations")
    for angle in angles:
        for i in range(len(angles_by_map[mapname][angle])):
            for j in range(angles_by_map[mapname][angle][i][0]):
                angles_by_map[mapname][angle][i][0][j] = (angles_by_map[mapname][angle][i][0][j][0] + translate[0], angles_by_map[mapname][angle][i][0][j][0] + translate[1])

    for angle in angles:
        print("Annotating angle " + str(angle))
        polygons = angles_by_map[mapname][angle]
	
        rot_image_name = mapname + "_" + str(angle) + ".tiff"

        if len(polygons) > 0:
            annotations.write(os.path.join(map_output_dir, rot_image_name) + "\n")
            annotations.write(str(len(polygons))+"\n")

            rot_img, rot_mat, bounds = rotate_image(map_img, angle, original_shape)
            for polygon, _ in polygons:
                #### Bounding boxes are given as coordinates on the original horizontal image ######
                #### I need to rotate the image, just as the network will see it as input, and determine the new coordinates of bounding boxes ####
                points = np.array([
                    np.asarray([ int(points[0]), int(point[1]) ]) for point in polygon
                ])

                ############# Visualize bounding boxes on original image ##################
                # cv2.drawContours(map_img, [points], 0, (255, 0, 0), 5)
                ###########################################################################

                transformed_points = rot_mat.dot(points.T).T

                ########## Visualize mapped bounding boxes after rotation ###############
                # uncomment the following two lines to visualize the bounding boxes 
                # in the new set of images
                #cv2.drawContours(rot_img, [transformed_points], 0, (255, 0, 0), 7)
                #########################################################################

                string = " ".join([ str(int(point[0])) + ',' + str(int(point[1]))  for point in transformed_points])
                annotations.write(string+"\n")

            cv2.imwrite(os.path.join(map_output_dir, rot_image_name), rot_img)

#        if os.path.isdir("./"+mapname) == False:
 #           os.mkdir("./"+mapname)

  #      cv2.imwrite("./"+mapname+"/"+str(angle)+".png", map_img)
