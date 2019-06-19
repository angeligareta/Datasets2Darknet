# Modification of the GTSRB script [http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Structure]

# Python program for converting the ppm files from The German Traffic Sign Recognition Benchmark (GTSRB) to jpg files
# in order to use them in YOLO. Besides, it generate a txt with all the paths to the converted images in darknet format.
# By Angel Igareta for SaferAuto [https://github.com/angeligareta/SaferAuto]
import csv
from common_config import *

# TO CHANGE
GTSDB_ROOT_PATH = "/media/angeliton/Backup1/DBs/Road Signs/GTSDB/"
RESIZE_PERCENTAGE = 0.6
DB_PREFIX = 'gtsdb-'


ANNOTATIONS_FILE_PATH = GTSDB_ROOT_PATH + "gt.txt"
INPUT_PATH = GTSDB_ROOT_PATH + "input-img/"  # Path to the ppm images of the GTSRB dataset.


def initialize_traffic_sign_classes():
    traffic_sign_classes.clear()
    traffic_sign_classes["0-prohibitory"] = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
    traffic_sign_classes["1-danger"] = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    traffic_sign_classes["2-mandatory"] = [33, 34, 35, 36, 37, 38, 39, 40]
    traffic_sign_classes["3-stop"] = [14]
    traffic_sign_classes["4-yield"] = [13]
    traffic_sign_classes["5-noentry"] = [17]
    traffic_sign_classes[str(OTHER_CLASS) + "-" + OTHER_CLASS_NAME] = [6, 12, 32, 41, 42]  # undefined, other, redbluecircles, diamonds


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

    object_class = int(row[5])
    object_class_adjusted = adjust_object_class(object_class)  # Adjust class category

    if (SHOW_IMG):
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


# Function for reading the images
def read_dataset(output_train_text_path, output_test_text_path, output_train_dir_path, output_test_dir_path):
    img_labels = {}  # Set of images and its labels [filename]: [()]
    update_db_prefix(DB_PREFIX)
    initialize_traffic_sign_classes()
    initialize_classes_counter()

    train_text_file = open(output_train_text_path, "a+")
    test_text_file = open(output_test_text_path, "a+")

    gt_file = open(ANNOTATIONS_FILE_PATH)  # Annotations file
    gt_reader = csv.reader(gt_file, delimiter=';')  # CSV parser for annotations file

    # WRITE ALL THE DATA IN A DICTIONARY (TO GROUP LABELS ON SAME IMG)
    for row in gt_reader:
        filename = row[0]
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
    total_false_negatives = len(total_false_negatives_dir.keys())
    max_false_data = round(total_annotated_images * TRAIN_PROB)  # False data: False negative + background

    print("total_false_negatives: " + str(total_false_negatives))
    print("total_annotated_images: " + str(total_annotated_images) + " == "
          + str(len(total_annotated_images_dir.keys())))
    print("max_false_data: " + str(max_false_data))

    # ADD FALSE IMAGES TO TRAIN
    if total_false_negatives > max_false_data:
        total_false_negatives = max_false_data

    if ADD_FALSE_DATA:
        add_false_negatives(total_false_negatives, total_false_negatives_dir, output_train_dir_path, train_text_file)

    #  max_imgs = 1000
    for filename in total_annotated_images_dir.keys():
        input_img_file_path = img_labels[filename][0]
        input_img = read_img(input_img_file_path)  # Read image from image_file_path
        input_img = resize_img_percentage(input_img, RESIZE_PERCENTAGE)  # Resize img
        input_img_labels = img_labels[filename][1:]

        # Get percentage for train and another for testing
        train_file = rand.choices([True, False], [TRAIN_PROB, TEST_PROB])[0]
        output_filename = DB_PREFIX + filename[:-4]

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


# read_dataset(OUTPUT_TRAIN_TEXT_PATH, OUTPUT_TEST_TEXT_PATH, OUTPUT_TRAIN_DIR_PATH, OUTPUT_TEST_DIR_PATH)
