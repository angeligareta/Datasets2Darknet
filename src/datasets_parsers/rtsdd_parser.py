# Modification of the GTSRB script [http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Structure]

# Python program for converting the ppm files from The German Traffic Sign Recognition Benchmark (GTSRB) to jpg files
# in order to use them in YOLO. Besides, it generate a txt with all the paths to the converted images in darknet format.
# By Angel Igareta for SaferAuto [https://github.com/angeligareta/SaferAuto]
import csv
from common_config import *

# TO CHANGE
RTSDD_ROOT_PATH = "/media/angeliton/Backup1/DBs/Road Signs/RTSD-D/"
RESIZE_PERCENTAGE = 0.6
DB_PREFIX = 'rtsdd-'


ANNOTATIONS_FILE_NAME = "full-gt.csv"
IMAGES_DIR_NAME = "rtsd-frames/"

def initialize_traffic_sign_classes():
    traffic_sign_classes.clear()
    # ", ".join(list(map(lambda x: '"' + str(x.split(".")[0] + '"'), os.listdir("."))))
    traffic_sign_classes["0-prohibitory"] = ["2_6", "3_10", "3_11_n", "3_12_n", "3_13_r", "3_14_r", "3_15_n", "3_16_n", "3_18", "3_18_2", "3_19", "3_2", "3_20", "3_21", "3_22", "3_23", "3_24_n", "3_25_n", "3_26", "3_27", "3_28", "3_29", "3_3", "3_30", "3_31", "3_32", "3_33", "3_4_1", "3_4_n", "3_5", "3_6", "3_7", "3_8", "3_9"]
    traffic_sign_classes["1-danger"] = ["1_23", "1_1", "1_10", "1_11", "1_11_1", "1_12", "1_12_2", "1_13", "1_14", "1_15", "1_16", "1_17", "1_18", "1_19", "1_2", "1_20", "1_20_2", "1_20_3", "1_21", "1_22", "1_24", "1_25", "1_26", "1_27", "1_28", "1_29", "1_30", "1_31", "1_32", "1_33", "1_5", "1_6", "1_7", "1_8", "1_9", "2_3", "2_3_2", "2_3_3", "2_3_4", "2_3_5", "2_3_6", "2_3_7"]
    traffic_sign_classes["2-mandatory"] = ["4_1_1", "4_1_2", "4_1_2_1", "4_1_2_2", "4_1_3", "4_1_4", "4_1_5", "4_1_6", "4_2_1", "4_2_2", "4_2_3", "4_3", "4_4", "4_5", "4_6_n", "4_7_n"]
    traffic_sign_classes["3-information"] = ["2_7", "5_11", "5_12", "5_14", "5_15_1", "5_15_2", "5_15_2_2", "5_15_3", "5_15_5", "5_15_7", "5_19_1", "5_20", "5_21", "5_22", "5_3", "5_4", "5_5", "5_6", "5_7_1", "5_7_2", "5_8", "6_15_1", "6_15_2", "6_15_3", "6_2_n", "6_3_1", "6_4", "6_6", "6_7", "6_8_1", "6_8_2", "6_8_3", "5_16", "5_17", "5_18", "7_1", "7_10", "7_11", "7_18", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7", "7_9"]
    traffic_sign_classes["4-stop"] = ["2_5"]
    traffic_sign_classes["5-yield"] = ["2_4"]
    traffic_sign_classes["6-noentry"] = ["3_1"]
    traffic_sign_classes[str(OTHER_CLASS) + "-" + OTHER_CLASS_NAME] = []


# It depends on the row format
def calculate_darknet_format(input_img, row):
    real_img_width, real_img_height = get_img_dim_plt(input_img)
    image_width = int(real_img_width * RESIZE_PERCENTAGE)
    image_height = int(real_img_height * RESIZE_PERCENTAGE)
    width_proportion = (real_img_width / image_width)
    height_proportion = (real_img_height / image_height)

    object_lb_x1 = float(row[1]) / width_proportion
    object_lb_y1 = float(row[2]) / height_proportion
    object_width = float(row[3]) / width_proportion
    object_height = float(row[4]) / height_proportion

    object_class = row[5]
    object_class_adjusted = adjust_object_class(object_class)  # Adjust class category

    if (SHOW_IMG):
        show_img(resize_img_plt(input_img, image_width, image_height), object_lb_x1, object_lb_y1, object_width, object_height)

    return parse_darknet_format(object_class_adjusted, image_width, image_height, 
                                object_lb_x1, object_lb_y1, object_lb_x1 + object_width, object_lb_y1 + object_height)


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

    annotations_file_path = RTSDD_ROOT_PATH + "/" + ANNOTATIONS_FILE_NAME
    images_dir_path = RTSDD_ROOT_PATH + "/" + IMAGES_DIR_NAME

    if os.path.isfile(annotations_file_path) & os.path.isdir(images_dir_path):
        gt_file = open(annotations_file_path)  # Annotations file
        gt_reader = csv.reader(gt_file, delimiter=',')
        next(gt_reader)

        # WRITE ALL THE DATA IN A DICTIONARY (TO GROUP LABELS ON SAME IMG)
        for row in gt_reader:
            filename = row[0]
            file_path = images_dir_path + filename

            if os.path.isfile(file_path):
                input_img = read_img_plt(file_path)
                darknet_label = calculate_darknet_format(input_img, row)
                object_class_adjusted = int(darknet_label.split()[0])

                if filename not in img_labels.keys():  # If it is the first label for that img
                    img_labels[filename] = [file_path]

                # Add only useful labels (not false negatives)
                if object_class_adjusted != OTHER_CLASS:
                    img_labels[filename].append(darknet_label)
        gt_file.close()
    else:
        print("In folder " + RTSDD_ROOT_PATH + " there are missing files. ")

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
    print("MAX FALSE DATA: " + str(max_false_data))

    # ADD FALSE IMAGES TO TRAIN
    if total_false_negatives > max_false_data:
        total_false_negatives = max_false_data

    if ADD_FALSE_DATA:
        add_false_negatives(total_false_negatives, total_false_negatives_dir, output_train_dir_path, train_text_file)

    #  max_imgs = 1000
    for filename in total_annotated_images_dir.keys():
        input_img_file_path = img_labels[filename][0]
        # Read image from image_file_path
        input_img = read_img(input_img_file_path)
        input_img = resize_img_percentage(input_img, RESIZE_PERCENTAGE)  # Resize img
        input_img_labels = img_labels[filename][1:]

        # Get percentage for train and another for testing
        train_file = rand.choices([True, False], [TRAIN_PROB, TEST_PROB])[0]
        output_filename = DB_PREFIX + filename[:-4]

        if train_file:
            write_data(output_filename, input_img, input_img_labels,
                       train_text_file, output_train_dir_path, train_file)
        else:
            write_data(output_filename, input_img, input_img_labels,
                       test_text_file, output_test_dir_path, train_file)

    train_text_file.close()
    test_text_file.close()

    return classes_counter_train, classes_counter_test
