# Program to extract img and labels from MASTIF and converting them to darknet format.
import csv
import re
import os.path
from common_config import *

# TO CHANGE
MASTIF_ROOT_PATH = "/media/angeliton/Backup1/DBs/Road Signs/MASTIF/"
RESIZE_PERCENTAGE = 0.9
DB_PREFIX = 'mastif-'


ANNOTATIONS_FOLDERS = ["TS2009", "TS2010", "TS2011"]
ANNOTATIONS_FILENAME = "index.seq"
INPUT_PATH = MASTIF_ROOT_PATH + "input-img/"


def initialize_traffic_sign_classes():
    traffic_sign_classes.clear()
    traffic_sign_classes["0-prohibitory"] = ["B03", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B13", "B14", "B15", "B16", "B17", "B18", "B19", "B20", "B21", "B22", "B23", "B24", "B25", "B26", "B27", "B28", "B29", "B30", "B31", "B32", "B33", "B34", "B35", "B36", "B37", "B38"]
    traffic_sign_classes["1-danger"] = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19", "A20", "A21", "A22", "A23", "A24", "A25", "A26", "A27", "A28", "A29", "A30", "A31", "A32", "A33", "A34", "A35", "A36", "A37", "A38", "A39", "A40", "A41", "A42", "A43", "A44", "A45", "A46"]
    traffic_sign_classes["2-mandatory"] = ["B44", "B45", "B46", "B47", "B48", "B49", "B50", "B51", "B52", "B53", "B54", "B55", "B56", "B57", "B58", "B59", "B60", "B61", "B62"]
    traffic_sign_classes["3-information"] = ["C01", "C02", "C03", "C05", "C06", "C10", "C29", "C30", "C31", "C32", "C33", "C34", "C35", "C36", "C37", "C38", "C39", "C40", "C41", "C42", "C43", "C44", "C45", "C46", "C47", "C48", "C49", "C50", "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58", "C59", "C60", "C61", "C62", "C63", "C64", "C65", "C68", "C69", "C70", "C71", "C72", "C73", "C75", "C77", "C86", "C88", "C89", "C90", "C91", "C92", "C93", "C96"]
    traffic_sign_classes["4-stop"] = ["B02"]
    traffic_sign_classes["5-yield"] = ["B01"]
    traffic_sign_classes["6-noentry"] = ["B04"]
    traffic_sign_classes[str(OTHER_CLASS) + "-" + OTHER_CLASS_NAME] = []  # undefined, other, redbluecircles, diamonds


# It depends on the row format
def calculate_darknet_format(input_img, row):
    real_img_width, real_img_height = get_img_dim_plt(input_img)
    image_width = int(real_img_width * RESIZE_PERCENTAGE)
    image_height = int(real_img_height * RESIZE_PERCENTAGE)
    width_proportion = (real_img_width / image_width)
    height_proportion = (real_img_height / image_height)

    x = float(row[2]) / width_proportion
    y = float(row[3]) / height_proportion
    w = float(row[4]) / width_proportion
    h = float(row[5]) / height_proportion

    object_class = row[1]
    object_class_adjusted = adjust_object_class(object_class)  # Adjust class category

    if SHOW_IMG:
        show_img(resize_img_plt(input_img, image_width, image_height), x, y, w, h)

    return parse_darknet_format(object_class_adjusted, image_width, image_height, x, y, x + w, y + h)


def add_file_to_dir(row, subfolder_name, img_labels):
    filename = row[0]
    file_path = INPUT_PATH + subfolder_name + "/" + filename

    if os.path.isfile(file_path):
        # If it is the first label for that img
        if filename not in img_labels.keys():  
            img_labels[subfolder_name + "-" + filename] = [file_path]

        # Loop for all the labels in the row and calculate darknet format every 5.
        while (len(row) > 1):
            input_img = read_img_plt(file_path)
            darknet_label = calculate_darknet_format(input_img, row)

            object_class_adjusted = int(darknet_label.split()[0])                   
            if object_class_adjusted != OTHER_CLASS:  # Add only useful labels (not false negatives)
                img_labels[subfolder_name + "-" + filename].append(darknet_label)
                # print("\t" + darknet_label)

            del row[1:6] # Remove 5 values from row (already seen)


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
        subfolder = INPUT_PATH + subfolder_name 
        subfolder_annotation_filename = subfolder + "/" + ANNOTATIONS_FILENAME
        if (os.path.exists(subfolder_annotation_filename)):
            subfolder_annotation_file = open(subfolder_annotation_filename, "r")
            # Format each line from mastif to csv format
            for line in subfolder_annotation_file.readlines():
                line = re.sub("[\[\]\(\)]", "", line)
                line = re.sub("[xywh]=", "", line)
                line = re.sub("[:&@,]", " ", line)
                # print("\t" + line)

                row = line.split(" ")
                add_file_to_dir(row, subfolder_name, img_labels)
            subfolder_annotation_file.close()                
        else:
            print("Subfolder " + subfolder + " not found")

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
    if total_false_negatives > max_false_data:
        total_false_negatives = max_false_data

    if ADD_FALSE_DATA:
        add_false_negatives(total_false_negatives, total_false_negatives_dir, output_train_dir_path, train_text_file)

    # SET ANNOTATED IMAGES IN TRAIN OR TEST DIRECTORIES
    for filename in total_annotated_images_dir.keys():
        input_img_file_path = img_labels[filename][0]
        input_img = read_img(input_img_file_path)  # Read image from image_file_path
        input_img = resize_img_percentage(input_img, RESIZE_PERCENTAGE)  # Resize img
        input_img_labels = img_labels[filename][1:]

        # Get percentage for train and another for testing
        train_file = rand.choices([True, False], [TRAIN_PROB, TEST_PROB])[0]
        output_filename = DB_PREFIX + filename

        if train_file:
            write_data(output_filename, input_img, input_img_labels, train_text_file, output_train_dir_path, train_file)
        else:
            write_data(output_filename, input_img, input_img_labels, test_text_file, output_test_dir_path, train_file)

    train_text_file.close()
    test_text_file.close()

    return classes_counter_train, classes_counter_test


# read_dataset()
