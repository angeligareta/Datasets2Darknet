# GENERAL PARSER: In this file you can import different dataset extractors with 
# the read_traffic_signs() method implemented, indicating how to read the dataset.
#
# The program will put those datasets together in a general one.
import datasets_parsers.btsdb_parser as BTSDB
import datasets_parsers.gtsdb_parser as GTSDB
import datasets_parsers.lisats_parser as LISATS


# Path where you want to save the dataset 
ROOT_PATH = "/home/angeliton/Desktop/SaferAuto/models/datasets/ere/"


# Path of the training and testing txt used as input for darknet.
OUTPUT_TRAIN_TEXT_PATH = ROOT_PATH + "train.txt"
OUTPUT_TEST_TEXT_PATH = ROOT_PATH + "test.txt"


# Path of the resulting training and testing images of this script and labels.
OUTPUT_TRAIN_DIR_PATH = ROOT_PATH + "output-img-train/"
OUTPUT_TEST_DIR_PATH = ROOT_PATH + "output-img-test/"


# Prints the object's number of each class of the received array.
def print_class_info(classes_counter):
    for i in range(0, len(classes_counter)):
        print('\t-CLASS: ' + str(i) + ' : ' + str(classes_counter[i]))
    print('TOTAL: ' + str(sum(classes_counter)))


# Prints the train classes, test classes and proportion train-test for a DB.
def print_db_info(classes_counter_train, classes_counter_test):
    print("[TRAIN FILES]")
    print_class_info(classes_counter_train)

    print("\n[TEST FILES]")
    print_class_info(classes_counter_test)

    print("\n[PROPORTION]")
    for i in range(0, len(classes_counter_test)):
        total_classes = classes_counter_train[i] + classes_counter_test[i]
        if total_classes == 0:
            total_classes = 1
        print('\t-CLASS: ' + str(i) + ' : ' + "{:.2f}%".format(classes_counter_test[i] / total_classes * 100.0))


# Given two arrays, returns the sum of them.
def add_arrays(array_1, array_2):
    for i in range(0, len(array_1)):
        array_2[i] += array_1[i]

    return array_2


# Main method. 
def main():
    print('GTSDB DATASET: ')
    classes_counter_train_total, classes_counter_test_total = \
        GTSDB.read_traffic_signs(OUTPUT_TRAIN_TEXT_PATH, OUTPUT_TEST_TEXT_PATH, OUTPUT_TRAIN_DIR_PATH, OUTPUT_TEST_DIR_PATH)
    print_db_info(classes_counter_train_total, classes_counter_test_total)

    print('BTSDB DATASET: ')
    classes_counter_train, classes_counter_test = \
          BTSDB.read_traffic_signs(OUTPUT_TRAIN_TEXT_PATH, OUTPUT_TEST_TEXT_PATH, OUTPUT_TRAIN_DIR_PATH, OUTPUT_TEST_DIR_PATH)
    print_db_info(classes_counter_train, classes_counter_test)
    classes_counter_train_total = add_arrays(classes_counter_train_total, classes_counter_train)
    classes_counter_test_total = add_arrays(classes_counter_test_total, classes_counter_test)

    print('LISATS DATASET: ')
    classes_counter_train, classes_counter_test = \
        LISATS.read_traffic_signs(OUTPUT_TRAIN_TEXT_PATH, OUTPUT_TEST_TEXT_PATH, OUTPUT_TRAIN_DIR_PATH, OUTPUT_TEST_DIR_PATH)
    print_db_info(classes_counter_train, classes_counter_test)
    classes_counter_train_total = add_arrays(classes_counter_train_total, classes_counter_train)
    classes_counter_test_total = add_arrays(classes_counter_test_total, classes_counter_test)

    print('TOTAL DATASET: ')
    print_db_info(classes_counter_train_total, classes_counter_test_total)


main()
