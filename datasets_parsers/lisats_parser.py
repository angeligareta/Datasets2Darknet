# Program to extract img and labels from LISATS and converting them to darknet format.

# Python program for converting the ppm files from The Belgium Traffic Sign dataset to jpg files
# in order to use them in YOLO. Besides, it generate a txt with all the paths to the converted images in darknet format.
# By Angel Igareta for SaferAuto [https://github.com/angeligareta/SaferAuto]
import csv
from common_config import *

LISATS_ROOT_PATH = "/media/angeliton/Backup1/DBs/Road Signs/LISATS/"

COMBINED_ANNOTATIONS_FILE_PATH = LISATS_ROOT_PATH + "allAnnotations.csv"

# Path to the ppm images of the BTSDB dataset.
INPUT_PATH = LISATS_ROOT_PATH + "input-img/"
BACKGROUND_IMG_PATH = LISATS_ROOT_PATH + "input-img-bg/"

# Path of the resulting training and testing images of this script and labels.

OUTPUT_TRAIN_DIR_PATH = LISATS_ROOT_PATH + "output-img-train/"
OUTPUT_TEST_DIR_PATH = LISATS_ROOT_PATH + "output-img-test/"

# Path of the training and testing txt used as input for darknet.
OUTPUT_TRAIN_TEXT_PATH = LISATS_ROOT_PATH + "train.txt"
OUTPUT_TEST_TEXT_PATH = LISATS_ROOT_PATH + "test.txt"

DB_PREFIX = 'lisats-'


def initialize_traffic_sign_classes():
    # Superclasses BTSDB
    traffic_sign_classes["0-prohibitory"] = []
    traffic_sign_classes["1-danger"] = []
    traffic_sign_classes["2-mandatory"] = []
    traffic_sign_classes["3-stop"] = ["stop"]
    traffic_sign_classes["4-yield"] = ["yield"]
    traffic_sign_classes["5-false_negatives"] = []


# It depends on the row format
def calculate_darknet_format(input_img, image_width, image_height, row):
    real_img_width, real_img_height = get_img_dim_plt(input_img)
    width_proportion = (real_img_width / MAX_WIDTH)
    height_proportion = (real_img_height / MAX_HEIGHT)

    left_x = float(row[2]) / width_proportion
    bottom_y = float(row[3]) / height_proportion
    right_x = float(row[4]) / width_proportion
    top_y = float(row[5]) / height_proportion

    object_class = row[1]
    object_class_adjusted = adjust_object_class(object_class)  # Adjust class category

    if SHOW_IMG:
        show_img(resize_img_plt(input_img), left_x, bottom_y, (right_x - left_x), (top_y - bottom_y))

    return parse_darknet_format(object_class_adjusted, image_width, image_height, left_x, bottom_y, right_x, top_y)


def read_dataset(output_train_text_path, output_test_text_path, output_train_dir_path, output_test_dir_path):
    img_labels = {}  # Set of images and its labels [filename]: [()]
    initialize_traffic_sign_classes()
    initialize_classes_counter()
    update_db_prefix(DB_PREFIX)

    train_text_file = open(output_train_text_path, "a+")
    test_text_file = open(output_test_text_path, "a+")

    gt_file = open(COMBINED_ANNOTATIONS_FILE_PATH)  # Annotations file
    gt_reader = csv.reader(gt_file, delimiter=';')  # CSV parser for annotations file

    # WRITE ALL THE DATA IN A DICTIONARY (TO GROUP LABELS ON SAME IMG)
    for row in gt_reader:
        filename = row[0].split("/")[-1][:-4]
        file_path = INPUT_PATH + row[0]

        if os.path.isfile(file_path):
            input_img = read_img_plt(file_path)
            darknet_label = calculate_darknet_format(input_img, MAX_WIDTH, MAX_HEIGHT, row)
            object_class_adjusted = int(darknet_label.split()[0])

            if filename not in img_labels.keys():  # If it is the first label for that img
                img_labels[filename] = [file_path]

            if object_class_adjusted != FALSE_NEGATIVE_CLASS:  # Add only useful labels (not false negatives)
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

    print('TOTAL ANNOTATED IMAGES: ' + str(len(total_annotated_images_dir.keys())))
    print('TOTAL FALSE NEGATIVES: ' + str(len(total_false_negatives_dir.keys())))

    # SET ANNOTATED IMAGES IN TRAIN OR TEST DIRECTORIES
    # max_imgs = 1000
    for filename in total_annotated_images_dir.keys():
        input_img_file_path = img_labels[filename][0]
        input_img = read_img(input_img_file_path)  # Read image from image_file_path
        input_img = resize_img(input_img)  # Resize img
        input_img_labels = img_labels[filename][1:]

        # Get percentage for train and another for testing
        train_file = rand.choices([True, False], [TRAIN_PROB, TEST_PROB])[0]
        output_filename = DB_PREFIX + filename

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
