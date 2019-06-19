# Modification of the GTSRB script [http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Structure]

# Python program for converting the ppm files from The German Traffic Sign Recognition Benchmark (GTSRB) to jpg files
# in order to use them in YOLO. Besides, it generate a txt with all the paths to the converted images in darknet format.
# By Angel Igareta for SaferAuto [https://github.com/angeligareta/SaferAuto]
import csv
from common_config import *

# TO CHANGE
GTSDB_ROOT_PATH = "/media/angeliton/Backup1/DBs/Road Signs/RTSD-C/"
RESIZE_PERCENTAGE = 1
DB_PREFIX = 'rtsd-'
# Supported class names
CLASS_NAMES = ["sl10sl", "sl20sl", "sl30sl", "sl40sl", "sl50sl", "sl60sl", "sl70sl", "sl80sl"]


INPUT_PATH = GTSDB_ROOT_PATH + "input-img/"  # Path to the ppm images of the GTSRB dataset.
ANNOTATIONS_FOLDERS = ["rtsd-r1", "rtsd-r3"]


def initialize_traffic_sign_classes(dif): # dif should be 2 in the rtsd-r3 dataset
    traffic_sign_classes.clear()
    traffic_sign_classes["0-sl10sl"] = [39 + dif]
    traffic_sign_classes["1-sl20sl"] = [40 + dif]
    traffic_sign_classes["2-sl30sl"] = [41 + dif]
    traffic_sign_classes["3-sl40sl"] = [42 + dif]
    traffic_sign_classes["4-sl50sl"] = [44 + dif]
    traffic_sign_classes["5-sl60sl"] = [45 + dif]
    traffic_sign_classes["6-sl70sl"] = [46 + dif]
    traffic_sign_classes["7-sl80sl"] = [47 + dif]
    traffic_sign_classes[str(OTHER_CLASS) + "-" + OTHER_CLASS_NAME] = []


def get_max_index(images_folder):
    if os.path.isdir(images_folder):
        images_names = os.listdir(images_folder)
        max_index = 0
        for name in images_names: 
            splitted_name = name.split("_")
            if(splitted_name[0].isdigit() & len(splitted_name) > 1):
                index = int(splitted_name[0])
                if(max_index < index):
                    max_index = index

        print(max_index)        
        return max_index
    else:
        raise Exception("Subfolder " + images_folder + "does not exist!")


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
    update_db_prefix(DB_PREFIX)
    initialize_traffic_sign_classes(0)
    initialize_classes_counter()

    # Important to not overwrite data!
    class_index_train = get_max_index(output_train_dir_path)
    class_index_test = get_max_index(output_test_dir_path)
    class_index = max([class_index_train, class_index_test]) + 1
    # print(CLASS_INDEX)

    train_text_file = open(output_train_text_path, "a+")
    test_text_file = open(output_test_text_path, "a+")
    train_text_file.write("\n")
    test_text_file.write("\n")

    for subfolder_name in ANNOTATIONS_FOLDERS:
        subfolder_path = INPUT_PATH + subfolder_name
        if os.path.isdir(subfolder_path):
            for subsubfolder_name in ["train", "test"]:
                annotation_filename = subfolder_path + "/gt_" + subsubfolder_name +".csv"
                subsubfolder_path = subfolder_path + "/" + subsubfolder_name
                if os.path.isfile(annotation_filename) & os.path.isdir(subfolder_path):
                    annotations = csv.reader(open(annotation_filename), delimiter=',')  # CSV parser for annotations file
                    next(annotations) # Skip header

                    for annotation in annotations:
                        filename, class_number = annotation
                        adjusted_class_name = get_object_label(int(class_number))
                        
                        if (adjusted_class_name != OTHER_CLASS_NAME):                            
                            image_path = subsubfolder_path + "/" + filename
                            image = read_img(image_path)
                            image = resize_img_percentage(image, RESIZE_PERCENTAGE)
                            # print(output_filename)

                            output_filename = str(class_index) + "_" + adjusted_class_name + OUTPUT_IMG_EXTENSION
                            train_file = rand.choices([True, False], [TRAIN_PROB, TEST_PROB])[0]
                            if train_file:
                                output_filename = output_train_dir_path + output_filename
                                train_text_file.write(output_filename + "\n")
                            else:
                                output_filename = output_test_dir_path + output_filename
                                test_text_file.write(output_filename + "\n")
                            
                            if SHOW_IMG:
                                print(output_filename)                        
                                cv2.imshow(output_filename, image)  
                                cv2.waitKey(3000)

                            cv2.imwrite(output_filename, image)
                            class_index += 1                             
                else:
                    print(annotation_filename + " or " + subsubfolder_path + "do not exist!")             
        else:
            print("The folder " + subfolder_path + "does not exist.")
        
        initialize_traffic_sign_classes(2) # For second dataset!

    return classes_counter_train, classes_counter_test


# read_dataset(OUTPUT_TRAIN_TEXT_PATH, OUTPUT_TEST_TEXT_PATH, OUTPUT_TRAIN_DIR_PATH, OUTPUT_TEST_DIR_PATH)
