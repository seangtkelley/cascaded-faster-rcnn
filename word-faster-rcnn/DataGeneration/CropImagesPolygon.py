import numpy as np 
import cv2
import sys
import os


crop_h = 512
crop_w = 512
step = 400

'''
crop_h = 100
crop_w = 100
step = 40
'''

curr_path = os.path.dirname(os.path.realpath(__file__))

annotations_filename = sys.argv[1]
annotations = open(annotations_filename, "r").readlines()

train_directory = sys.argv[2]

cropped_annotations = open(os.path.join(train_directory, "cropped_annotations_angles_polygon.txt"), "w")

cropped_img_output_dir = os.path.join(os.sep, 'mnt', 'nfs', 'work1', 'elm', 'sgkelley', 'data', 'maps', 'angles_-90to90step5', 'windows')

if os.path.isdir(cropped_img_output_dir) == False:
    os.mkdir(cropped_img_output_dir)

def window_contains_polygon(current_x, current_y, crop_h, crop_w, polygon):
    for point in polygon:
        if point[0] > current_x and  point[0] < (current_x+crop_w) and point[1] < (current_y+crop_h) and point[1] > current_y:
            return False
    return True

def create_crops(img, img_name, regions):
    print(img_name)
    height = img.shape[0]; width = img.shape[1]
    current_x = 0; current_y = 0
    index = 0

    print(img.shape)
    print("Curr y " + str(current_y) + " - crop_h " + str(crop_h) + " height " + str(height))
    print("Curr x " + str(current_x) + " - crop_w " + str(crop_w) + " width " + str(width))
    print("============================================")
    
    while current_y + crop_h < height:
        while current_x + crop_w < width:
            crop_img = img[current_y:current_y+crop_h, current_x:current_x+crop_w]

            split_img = img_name.split('/')[-1].split(".")
            cropped_img_name = os.path.join(cropped_img_output_dir, split_img[0]+"_"+str(current_x)+"x"+str(current_y)+"."+split_img[1])

            cropped_regions = []

            for polygon in regions:

                #### Height (h) moves up in index.
                if window_contains_polygon(current_x, current_y, crop_h, crop_w, polygon):
                    translated_polygon = polygon
                    for i in range(len(polygon)):
                        translated_polygon[i][0] = polygon[i][0] - current_x
                        translated_polygon[i][1] = polygon[i][1] - current_y

                    cropped_regions.append( translated_polygon )

                    # uncomment the following to print the actual bounding boxes
                    # cv2.drawContours(crop_img, [translated_polygon], 0, (255, 0, 0), 2)

            cv2.imwrite(cropped_img_name, crop_img)
            if len(cropped_regions) > 0:
                cropped_annotations.write(cropped_img_name+"\n")
                cropped_annotations.write(str(len(cropped_regions))+"\n")
                for r in cropped_regions:
                    cropped_annotations.write( " ".join([ str(int(point[0])) + ',' + str(int(point[1]))  for point in r]) + "\n")

            index += 1
            current_x += step

        current_x = 0
        current_y += step



image = None
image_path = "" 
regions = None
count = 0
for line in annotations:
    if line.endswith(".tiff\n"):
        count += 1
        if regions is not None:
            create_crops(image, image_path, regions)

        image_path = line.replace("\n", "")
        print("Reading image: " + image_path)
        image = cv2.imread(image_path)
        print(image.shape)
        regions = []

    elif len(line.split(" ")) >= 4:
         split_line = line.split(" ")

        points = [ [int(point.split(',')[0]), int(point.split(',')[1])] for point in split_line ]

        regions.append( points )

print(image_path)
print("Images: " + str(count))
create_crops(image, image_path, regions)

