# GENERAL PARSER: In this file you can import different dataset extractors with 
# the read_dataset() method implemented, indicating how to read the dataset.
#
# The program will put those datasets together in a general one.
import click

import datasets_parsers.gtsdb_parser as GTSDB
import datasets_parsers.btsdb_parser as BTSDB
import datasets_parsers.lisats_parser as LISATS
import datasets_parsers.lisatl_parser as LISATL
import datasets_parsers.mastif_parser as MASTIF
import datasets_parsers.rtsdc_parser as RTSDC
import datasets_parsers.rtsdd_parser as RTSDD
from common_config import *


# Datasets to use
DATASETS = [LISATL, MASTIF, RTSDD]
DATASETS_NAMES = ["LISATL", "MASTIF", "RTSDD"]


# Main method.
@click.command()
@click.option('--root_path', default="/home/angeliton/Desktop/SaferAuto/models/datasets/ere/", help='Path where you want to save the dataset.')
@click.option('--train_pct', default=TRAIN_PROB, help='Percentage of train images in final dataset. Format (0.0 - 1.0)')
@click.option('--test_pct', default=TEST_PROB, help='Percentage of test images in final dataset. Format (0.0 - 1.0)')
@click.option('--color_mode', default=COLOR_MODE, help='OpenCV Color mode for reading the images. (-1 (default) => color, 0 => bg).')
@click.option('--output_img_ext', default=OUTPUT_IMG_EXTENSION, help='Extension for output images. Default => .jpg')
@click.option('--verbose', is_flag=True, help='Option to show images while reading them.')
@click.option('--false_data', is_flag=True, help='Option for adding false data from datasets parsers if available.')
def main(root_path, train_pct, test_pct, color_mode, verbose, false_data, output_img_ext):
    # Path of the training and testing txt used as input for darknet.
    if (root_path[-1] != '/'):
        root_path += "/"
    output_train_text_path = root_path + "train.txt"
    output_test_text_path = root_path + "test.txt"
    # Path of the resulting training and testing images of this script and labels.
    output_train_dir_path = root_path + "train/"
    output_test_dir_path = root_path + "test/"

    classes_counter_train_total = classes_counter_train.copy()
    classes_counter_test_total = classes_counter_test.copy()        
    for dataset_index in range(0, len(DATASETS)):
        print(DATASETS_NAMES[dataset_index] + ' DATASET: ')

        # Update the dataset variables
        DATASETS[dataset_index].update_global_variables(train_pct, test_pct, color_mode, verbose, false_data, output_img_ext)

        # Read dataset
        classes_counter_train_partial, classes_counter_test_partial = \
            DATASETS[dataset_index].read_dataset(output_train_text_path, output_test_text_path, output_train_dir_path, output_test_dir_path)
        classes_counter_train_total = add_arrays(classes_counter_train_total, classes_counter_train_partial)
        classes_counter_test_total = add_arrays(classes_counter_test_total, classes_counter_test_partial)        
        
        print_db_info(classes_counter_train_partial, classes_counter_test_partial)

    print('TOTAL DATASET: ')
    print_db_info(classes_counter_train_total, classes_counter_test_total)


main()
