import numpy as np 
import cv2
import sys
import os


crop_h = 300
crop_w = 300
step = 100

'''
crop_h = 100
crop_w = 100
step = 40
'''
annotations_filename = sys.argv[1]
annotations = open(annotations_filename, "r").readlines()

images_directory = sys.argv[2]
train_directory = sys.argv[3]

cropped_annotations = open(train_directory+"cropped_annotations.txt", "w")

if os.path.isdir("./cropped_img") == False:
    os.mkdir("./cropped_img")

def create_crops(img, img_name, regions):
    print(img_name)
    height = img.shape[0]; width = img.shape[1]
    current_x = 0; current_y = 0
    index = 0

    print(img.shape)
    print("Curr y " + str(current_y) + " - crop_h " + str(crop_h) + " height " + str(height))
    print("Curr x " + str(current_x) + " - crop_w " + str(crop_w) + " width" + str(width))
    print("============================================")
    
    while current_y + crop_h < height:
        while current_x + crop_w < width:
            crop_img = img[current_y:current_y+crop_h, current_x:current_x+crop_w]

            split_img = img_name.split(".")
            cropped_img_name = "./cropped_img/"+split_img[0]+"_"+str(index)+"."+split_img[1]

            cropped_regions = []

            for region in regions:
                x, y, w, h = region
                #### Height (h) moves up in index.
                if x > current_x and  x+w < (current_x+crop_w) and y < (current_y+crop_h) and y+h > current_y:
                    crop_x = x - current_x
                    crop_y = y - current_y

                    cropped_regions.append( (crop_x, crop_y, w, h) )

                    cnt = np.array([[int(crop_x), int(crop_y)], [int(crop_x+w), int(crop_y)],
                                    [int(crop_x+w), int(crop_y+h)], [int(crop_x), int(crop_y+h)]])
		    # uncomment the following to print the actual bounding boxes
                    # cv2.drawContours(crop_img, [cnt], 0, (255, 0, 0), 2)

            cv2.imwrite(cropped_img_name, crop_img)
            if len(cropped_regions) > 0:
                cropped_annotations.write(cropped_img_name+"\n")
                cropped_annotations.write(str(len(cropped_regions))+"\n")
                for r in cropped_regions:
                    cropped_annotations.write( str(r[0]) + " " + str(r[1]) + " " + str(r[2]) + " " + str(r[3]) + "\n")

            index += 1
            current_x += step

        current_x = 0
        current_y += step



image = None
image_name = "" 
regions = None
count = 0
for line in annotations:
    if line.endswith(".tiff\n"):
        count += 1
        if regions is not None:
            create_crops(image, image_name, regions)

        image_name = line.split("/")[1][:-1]
        print("Reading image: " + images_directory+image_name)
        image = cv2.imread(images_directory+image_name)
        print(image.shape)
        regions = []
    elif len(line.split(" ")) == 4:
        split_line = line.split(" ")
        x = float(split_line[0]); y = float(split_line[1])
        r_w = float(split_line[2]); r_h = float(split_line[3])

        regions.append( (x, y, r_w, r_h) )

print(image_name)
print("Images: " + str(count))
create_crops(image, image_name, regions)

