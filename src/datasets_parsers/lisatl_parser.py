# Program to extract img and labels from MASTIF and converting them to darknet format.
import csv
import re
import os.path
from common_config import *

# TO CHANGE
INPUT_PATH = "/media/angeliton/Backup1/DBs/Traffic Light/LISATL/"
RESIZE_PERCENTAGE = 0.45
DB_PREFIX = 'lisatl-'

ANNOTATIONS_FOLDERS = ["dayTrain"]
ANNOTATIONS_FILENAME = "frameAnnotationsBOX.csv"


def initialize_traffic_sign_classes():
    traffic_sign_classes.clear()
    traffic_sign_classes["7-tlred"] = ["stop", "stopLeft"]
    traffic_sign_classes["8-tlamber"] = ["warning", "warningLeft"] 
    traffic_sign_classes["9-tlgreen"] = ["go", "goForward", "goLeft"]
    traffic_sign_classes[str(OTHER_CLASS) + "-" + OTHER_CLASS_NAME] = []


# It depends on the row format
def calculate_darknet_format(input_img,row):
    real_img_width, real_img_height = get_img_dim_plt(input_img)
    image_width = int(real_img_width * RESIZE_PERCENTAGE)
    image_height = int(real_img_height * RESIZE_PERCENTAGE)
    width_proportion = (real_img_width / image_width)
    height_proportion = (real_img_height / image_height)

    left_x = float(row[2]) / width_proportion
    bottom_y = float(row[3]) / height_proportion
    right_x = float(row[4]) / width_proportion
    top_y = float(row[5]) / height_proportion

    object_class = row[1]
    object_class_adjusted = adjust_object_class(object_class)  # Adjust class category

    if SHOW_IMG:
        show_img(resize_img_plt(input_img, image_width, image_height), left_x, bottom_y, (right_x - left_x), (top_y - bottom_y))

    return parse_darknet_format(object_class_adjusted, image_width, image_height, left_x, bottom_y, right_x, top_y)


def add_file_to_dir(row, subfolder_path, img_labels):
    filename = row[0].split("/")[1]
    file_path = subfolder_path + "/frames/" + filename

    if os.path.isfile(file_path):
        # If it is the first label for that img add it
        if filename not in img_labels.keys():  
            img_labels[filename] = [file_path]
        
        # Calculate darknet_format
        input_img = read_img_plt(file_path)
        darknet_label = calculate_darknet_format(input_img, row)

        object_class_adjusted = int(darknet_label.split()[0])                   
        if object_class_adjusted != OTHER_CLASS:  # Add only useful labels (not false negatives)
            img_labels[filename].append(darknet_label)
            # print("\t" + str(img_labels[filename]))
    else: 
        print("Image " + file_path + " not found")


def update_global_variables(train_pct, test_pct, color_mode, verbose, false_data, output_img_ext):
    global TRAIN_PROB, TEST_PROB, COLOR_MODE, SHOW_IMG, ADD_FALSE_DATA, OUTPUT_IMG_EXTENSION
    TRAIN_PROB = train_pct
    TEST_PROB = test_pct
    COLOR_MODE = color_mode
    SHOW_IMG = verbose
    ADD_FALSE_DATA = false_data
    OUTPUT_IMG_EXTENSION = output_img_ext
    

def read_dataset(output_train_text_path, output_test_text_path, output_train_dir_path, output_test_dir_path):
    img_labels = {}  # Set of images and its labels [filename]: [()]
    update_db_prefix(DB_PREFIX)
    initialize_traffic_sign_classes()
    initialize_classes_counter()

    train_text_file = open(output_train_text_path, "a+")
    test_text_file = open(output_test_text_path, "a+")

    # Loop between datasets subfolders
    for subfolder_name in ANNOTATIONS_FOLDERS:
        subfolder_path = INPUT_PATH + subfolder_name         
        # print(subfolder_path)

        for subsubfolder_name in os.listdir(subfolder_path):
            subsubfolder_path = subfolder_path + "/" + subsubfolder_name
            if os.path.isdir(subsubfolder_path):
                # print("Fetching data from " + subsubfolder_name + "...")
                subfolder_annotation_filename = subsubfolder_path + "/" + ANNOTATIONS_FILENAME

                # Check if annotation file exists
                if (os.path.exists(subfolder_annotation_filename)):
                    subfolder_annotation_file = open(subfolder_annotation_filename, "r")

                    for line in subfolder_annotation_file.readlines()[1:]: # Remove header
                        # print("\t" + line)
                        row = line.split(";")
                        add_file_to_dir(row, subsubfolder_path, img_labels)
                    subfolder_annotation_file.close()                
                else:
                    print("Subfolder " + subfolder_annotation_filename + " not found")

    # COUNT FALSE NEGATIVES (IMG WITHOUT LABELS)
    total_false_negatives_dir = {}
    total_annotated_images_dir = {}
    for filename in img_labels.keys():
        img_label_subset = img_labels[filename]
        if len(img_label_subset) == 1:
            total_false_negatives_dir[filename] = img_label_subset
        else:
            total_annotated_images_dir[filename] = img_label_subset

    # CALCULATE MAXIMUM FALSE NEGATIVES TO ADD
    total_annotated_images = len(img_labels.keys()) - len(total_false_negatives_dir.keys())
    total_false_negatives = len(total_false_negatives_dir.keys())
    max_false_data = round(total_annotated_images * TRAIN_PROB)  # False data: False negative + background

    print("TOTAL ANNOTATED IMAGES: " + str(total_annotated_images))
    print("TOTAL FALSE NEGATIVES: " + str(total_false_negatives))
    print("MAX FALSE DATA: " + str(max_false_data))

    # ADD FALSE IMAGES TO TRAIN
    #if total_false_negatives > max_false_data:
    #    total_false_negatives = max_false_data
    # add_false_negatives(total_false_negatives, total_false_negatives_dir, output_train_dir_path, train_text_file)

    # SET ANNOTATED IMAGES IN TRAIN OR TEST DIRECTORIES
    for filename in total_annotated_images_dir.keys():
        input_img_file_path = img_labels[filename][0]
        input_img = read_img(input_img_file_path)  # Read image from image_file_path
        input_img = resize_img_percentage(input_img, RESIZE_PERCENTAGE)  # Resize img
        input_img_labels = img_labels[filename][1:]

        # Get percentage for train and another for testing
        train_file = rand.choices([True, False], [TRAIN_PROB, TEST_PROB])[0]
        output_filename = DB_PREFIX + filename[:-4] # Remove .png

        if train_file:
            write_data(output_filename, input_img, input_img_labels, train_text_file, output_train_dir_path, train_file)
        else:
            write_data(output_filename, input_img, input_img_labels, test_text_file, output_test_dir_path, train_file)

    train_text_file.close()
    test_text_file.close()

    return classes_counter_train, classes_counter_test


# read_dataset()
