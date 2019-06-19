# Program to extract img and labels from BTSDB and converting them to darknet format.

# Python program for converting the ppm files from The Belgium Traffic Sign dataset to jpg files
# in order to use them in YOLO. Besides, it generate a txt with all the paths to the converted images in darknet format.
# By Angel Igareta for SaferAuto [https://github.com/angeligareta/SaferAuto]
import csv
from common_config import *

# TO CHANGE
BTSDB_ROOT_PATH = "/media/angeliton/Backup1/DBs/Road Signs/BTSDB/"
RESIZE_PERCENTAGE = 0.4
DB_PREFIX = 'btsdb-'


COMBINED_ANNOTATIONS_FILE_PATH = BTSDB_ROOT_PATH + "annotations-combined.txt"

# Path to the ppm images of the BTSDB dataset.
INPUT_PATH = BTSDB_ROOT_PATH + "input-img/"
BACKGROUND_IMG_PATH = BTSDB_ROOT_PATH + "input-img-bg/"


def initialize_traffic_sign_classes():
    traffic_sign_classes.clear()
    # Superclasses BTSDB
    traffic_sign_classes["0-prohibitory"] = [2, 8]
    traffic_sign_classes["1-danger"] = [1]
    traffic_sign_classes["2-mandatory"] = [3]
    traffic_sign_classes["3-stop"] = [7]
    traffic_sign_classes["4-yield"] = [6]
    traffic_sign_classes["5-noentry"] = [42]
    traffic_sign_classes[str(OTHER_CLASS) + "-" + OTHER_CLASS_NAME] = [-1, 0, 4, 5, 9, 10, 11]  # undefined, other, redbluecircles, diamonds


# It depends on the row format
def calculate_darknet_format(input_img, row):
    real_img_width, real_img_height = get_img_dim_plt(input_img)
    image_width = int(real_img_width * RESIZE_PERCENTAGE)
    image_height = int(real_img_height * RESIZE_PERCENTAGE)
    width_proportion = (real_img_width / image_width)
    height_proportion = (real_img_height / image_height)

    left_x = float(row[1]) / width_proportion
    bottom_y = float(row[2]) / height_proportion
    right_x = float(row[3]) / width_proportion
    top_y = float(row[4]) / height_proportion

    specific_object_class = int(row[5])
    if (specific_object_class == 42): # Save noentry signs
        object_class = specific_object_class
    else:
        object_class = int(row[6])
    object_class_adjusted = adjust_object_class(object_class)  # Adjust class category

    if SHOW_IMG:
        show_img(resize_img_plt(input_img, image_width, image_height), left_x, bottom_y, (right_x - left_x), (top_y - bottom_y))

    return parse_darknet_format(object_class_adjusted, image_width, image_height, left_x, bottom_y, right_x, top_y)


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

    gt_file = open(COMBINED_ANNOTATIONS_FILE_PATH)  # Annotations file
    gt_reader = csv.reader(gt_file, delimiter=';')  # CSV parser for annotations file

    # WRITE ALL THE DATA IN A DICTIONARY (TO GROUP LABELS ON SAME IMG)
    for row in gt_reader:
        filename = row[0][:-4] + ".jpg"  # Replace jp2 with jpg
        file_path = INPUT_PATH + filename

        if os.path.isfile(file_path):
            input_img = read_img_plt(file_path)
            darknet_label = calculate_darknet_format(input_img, row)
            object_class_adjusted = int(darknet_label.split()[0])

            if filename not in img_labels.keys():  # If it is the first label for that img
                img_labels[filename] = [file_path]

            if object_class_adjusted != OTHER_CLASS:  # Add only useful labels (not false negatives)
                img_labels[filename].append(darknet_label)

    # COUNT FALSE NEGATIVES (IMG WITHOUT LABELS)
    total_false_negatives_dir = {}
    total_annotated_images_dir = {}
    for filename in img_labels.keys():
        img_label_subset = img_labels[filename]
        if len(img_label_subset) == 1:
            total_false_negatives_dir[filename] = img_label_subset
        else:
            total_annotated_images_dir[filename] = img_label_subset

    total_annotated_images = len(img_labels.keys()) - len(total_false_negatives_dir.keys())
    total_false_data = round(total_annotated_images * TRAIN_PROB)  # False data: False negative + background

    print("total_false_negatives: " + str(len(total_false_negatives_dir.keys())))
    print("total_annotated_images: " + str(total_annotated_images) + " == "
          + str(len(total_annotated_images_dir.keys())))
    print("total_false_data: " + str(total_false_data))

    # ADD FALSE IMAGES TO TRAIN
    if ADD_FALSE_DATA:
        add_false_data(total_false_data, total_false_negatives_dir, BACKGROUND_IMG_PATH, output_train_dir_path, train_text_file)

    # SET ANNOTATED IMAGES IN TRAIN OR TEST DIRECTORIES
    # max_imgs = 1000
    for filename in total_annotated_images_dir.keys():
        input_img_file_path = img_labels[filename][0]
        input_img = read_img(input_img_file_path)  # Read image from image_file_path
        input_img = resize_img_percentage(input_img, RESIZE_PERCENTAGE)  # Resize img
        input_img_labels = img_labels[filename][1:]

        # Get percentage for train and another for testing
        train_file = rand.choices([True, False], [TRAIN_PROB, TEST_PROB])[0]
        output_filename = DB_PREFIX + filename[3:-4]

        if train_file:
            write_data(output_filename, input_img, input_img_labels, train_text_file, output_train_dir_path, train_file)
        else:
            write_data(output_filename, input_img, input_img_labels, test_text_file, output_test_dir_path, train_file)

        # max_imgs -= 1
        # if max_imgs == 0:
        #    break

    gt_file.close()
    train_text_file.close()
    test_text_file.close()

    return classes_counter_train, classes_counter_test

# read_dataset()
