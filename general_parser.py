# GENERAL PARSER: In this file you can import different dataset extractors with 
# the read_dataset() method implemented, indicating how to read the dataset.
#
# The program will put those datasets together in a general one.
import datasets_parsers.gtsdb_parser as GTSDB
import datasets_parsers.btsdb_parser as BTSDB
import datasets_parsers.lisats_parser as LISATS
import datasets_parsers.mastif_parser as MASTIF
from common_config import *

# Path where you want to save the dataset 
ROOT_PATH = "D:/Data/Desktop/datasets/ere/"

# Path of the training and testing txt used as input for darknet.
OUTPUT_TRAIN_TEXT_PATH = ROOT_PATH + "train.txt"
OUTPUT_TEST_TEXT_PATH = ROOT_PATH + "test.txt"

# Path of the resulting training and testing images of this script and labels.
OUTPUT_TRAIN_DIR_PATH = ROOT_PATH + "output-img-train/"
OUTPUT_TEST_DIR_PATH = ROOT_PATH + "output-img-test/"

# Datasets to use
DATASETS = [GTSDB, BTSDB, LISATS, MASTIF] 

# Main method. 
def main():
    classes_counter_train_total = []
    classes_counter_test_total = []
    
    for dataset in DATASETS:
        classes_counter_train, classes_counter_test = \
            dataset.read_dataset(OUTPUT_TRAIN_TEXT_PATH, OUTPUT_TEST_TEXT_PATH, OUTPUT_TRAIN_DIR_PATH, OUTPUT_TEST_DIR_PATH)
        classes_counter_train_total = add_arrays(classes_counter_train_total, classes_counter_train)
        classes_counter_test_total = add_arrays(classes_counter_test_total, classes_counter_test)
        
        print(DB_PREFIX[:-1] + ' DATASET: ')
        print_db_info(classes_counter_train, classes_counter_test)

    print('TOTAL DATASET: ')
    print_db_info(classes_counter_train_total, classes_counter_test_total)


main()
