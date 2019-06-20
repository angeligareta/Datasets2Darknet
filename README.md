<h1 align="center">Datasets2Darknet</h1>
<h4 align="center">Modular tool that extracts images and labels from multiple datasets and parses them to Darknet format. </h4>

<p align="center">
  <a href="https://github.com/angeligareta/Datasets2Darknet/blob/master/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/angeligareta/Datasets2Darknet.svg?style=for-the-badge">
  </a>
  <a href="https://github.com/ellerbrock/open-source-badges/">
    <img alt="Website" src="https://badges.frapsoft.com/os/v1/open-source-175x29.png?v=103">
  </a>
</p>

<hr/>

Datasets2Darknet allows you to merge multiple datasets into one while converting them to Darknet format. It is very modular, easing the process of adding new datasets.

## Current available datasets
The idea of this section is to add parsers for new object datasets, with the aim of supporting the unification of the maximum possible number of different datasets. Darknet labels vary depending on the task. The labels for Detection Task *(./darknet detector)* are not the same that the ones for Classification Task *(./darknet classifier)*. 

For the moment, in the [dataset_parsers](./src/datasets_parsers/) folder there are available the following datasets.

### Detection Task
#### Traffic Sign Datasets
- [German Traffic Sign Detection Benchmark](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset) - Dataset Parser at [src/datasets_parsers/gtsdb_parser](./src/datasets_parsers/gtsdb_parser.py)
- [Belgium Traffic Sign Dataset:](https://btsd.ethz.ch/shareddata/) - Dataset Parser at [src/datasets_parsers/btsdb_parser](./src/datasets_parsers/btsdb_parser.py)
- [Mapping and Assessing the State of Traffic InFrastructure (MASTIF) Dataset](http://www.zemris.fer.hr/~ssegvic/mastif/datasets.shtml) - Dataset Parser at [src/datasets_parsers/mastif_parser](./src/datasets_parsers/mastif_parser.py)
- [LISA Traffic Sign Dataset](http://cvrr.ucsd.edu/vivachallenge/index.php/signs/sign-detection/) - Dataset Parser at [src/datasets_parsers/lisats_parser](./src/datasets_parsers/lisats_parser.py)
- [Russian Traffic Sign Dataset](http://graphics.cs.msu.ru/en/research/projects/rtsd) - Dataset Parser at [src/datasets_parsers/rtsdd_parser](./src/datasets_parsers/rtsdd_parser.py)
#### Traffic Light Datasets
- [LISA Traffic Light Dataset](http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html) - Dataset Parser at [src/datasets_parsers/lisatl_parser](./src/datasets_parsers/lisatl_parser.py)

### Classification Task
- [Russian Traffic Sign Dataset](http://graphics.cs.msu.ru/en/research/projects/rtsd) - Dataset Parser at [src/datasets_parsers/rtsdc_parser](./src/datasets_parsers/rtsdc_parser.py)

## Main files 
### Common Config (common_config.py)
All the common methods for the specific dataset parsers are contained in this file, for instance: read_image, resize_image, write_data and so on. Feel free to check them out, each one is documented. 

On top of that, there are several constants that you can change according to your preferences. These are:
- **TRAIN_PROB, TEST_PROB**: Percentage of train-test proportion for the input images.
- **OUTPUT_IMG_EXTENSION**: Extension of the output images. (Default: jpg)
- **COLOR MODE**: Default -1 (RGB). If you want to read the images in black-white scale use 0 option.
- **SHOW_IMG**: If activated it will show each processed image and the annotated bounding boxes in it.
- **ADD_FALSE_DATA**: If activated it will add the false data with a blank txt file as background file for training.


### General parser (general_parser.py) 
Main file of the program. It imports all the specific datasets and loop over them calling the read_dataset method that returns the count of the classes read in the specific dataset. 

At the end, it shows the total number of annotated images per class and train-test proportion.

### Datasets Parsers
Directory that contains all the specific datasets parsers. 

## How to use available datasets
In order to convert the labels of one of the [current available datasets](https://github.com/angeligareta/Datasets2Darknet#current-available-datasets) to Darknet Format, you need to follow these steps.

### 1º Modify the output path of the unified dataset
In the [src/general_parser](./src/general_parser.py) you must specify the path where the output images and labels will be stored. This can be done by easily changing the variable named ROOT_PATH. The files for train and test image paths and the folders for train and test images and annotations will be created using that path as base.

### 2º Specify the datasets to use
Once you have selected the dataset parsers you are going to use from the [current available datasets](https://github.com/angeligareta/Datasets2Darknet#current-available-datasets), you have to import them in the [src/general_parser](./src/general_parser.py) file. For example, for importing the German Traffic Sign Detection Benchmark and the MASTIF dataset you would need to add:
```python
import datasets_parsers.gtsdb_parser as GTSDB
import datasets_parsers.mastif_parser as MASTIF
```
Now, you simply need to add the datasets you want to convert annotations from in the DATASETS variable, as well as their names in DATASETS_NAMES. For extracting data from GTSDB and MASTIF dataset and save the result in ROOT_PATH, these variables would need to have the following values:
```python
DATASETS = [GTSDB, MASTIF]
DATASETS_NAMES = ["GTSDB", "MASTIF"]
```
### 3º Modify the input folders from the parsers file
Once you have downloaded the images and annotations from the datasets you are going to use, you should extract the information to separate folders. After that, the last step would be to modify the specific parsers you selected. You need to modify the paths contained in the file and adjust them to the location of these information in your computer. 

For example, if we have the [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/) dataset downloaded at "/home/angeliton/Desktop/DBs/Road Signs/GTSDB/", you should modfiy the GTSDB_ROOT_PATH variable at [src/datasets_parsers/gtsdb_parser](./src/datasets_parsers/gtsdb_parser.py) to that path.

### 4º Run the general parser
Finally, you just need to execute the general parser python program. From the root path you have to execute:
```
python3 general_parser.py
```

## How to add a new dataset
### 1º Add your dataset parser
As each dataset has specific annotation formats, we need a specific parser for each one. However, most of the methods are common, so the process is not difficult.

The methods the dataset parser must have are:
- *initialize_traffic_sign_classes()*: This method creates the relations between the real class id of an object in the dataset and the class id we are using for that object in the program. For instance, if a yield object has the class id "A02", we would need to add a relation such as:
```python 
def initialize_traffic_sign_classes():
    traffic_sign_classes["4-yield"] = [6]
```
- *calculate_darknet_format (input_img, image_width, image_height, row)*: This method converts the specific annotation format for a dataset to the darknet format. First of all we calculate the width and height proportion for the image. After that, we need to retrieve the bounding boxes borders from the specific dataset row and calculate the new positions according to the width and height proportions. Finallu we use the parse_darknet_format common method that needs the left_x, bottom_y, right_x and top_y values. 
```python
def calculate_darknet_format(input_img, image_width, image_height, row):
    real_img_width, real_img_height = get_img_dim_plt(input_img)
    img_width = int(real_img_width * RESIZE_PERCENTAGE)
    img_height = int(real_img_height * RESIZE_PERCENTAGE)

    width_proportion = (real_img_width / img_width)
    height_proportion = (real_img_height / img_height)

    object_lb_x1 = float(row[1]) / width_proportion
    object_lb_y1 = float(row[2]) / height_proportion
    object_width = float(row[3]) / width_proportion
    object_height = float(row[4]) / height_proportion

    obj_class = int(row[5])
    adjusted_obj_class = adjust_object_class(obj_class)  # Adjust class category

    if SHOW_IMG:
        show_img(resize_img_plt(input_img, img_width, img_height), 
                left_x, bottom_y, (right_x - left_x), (top_y - bottom_y))

    return parse_darknet_format(adjusted_obj_class, img_width, img_height, left_x, bottom_y, right_x, top_y)
```
- *read_dataset(output_train_text_path, output_test_text_path, output_train_dir_path, output_test_dir_path):* This is the main method. It reads the annotations file or files of the specific dataset, parse them to darknet format through the previous method and write them in the output paths received by argument. You can see an example of how to read different datasets in the [datasets_parsers](datasets_parsers/) folder but this really depends on the datasets way of organizing the images and annotations.

### 2º Add the dataset parser to the general parser
The second process is very easy. At the beginning of the general parser, you have to import the new dataset parser you have just created. For example, if we have a new dataset parser called btsdb_parser we would do: 
```python
import datasets_parsers.btsdb_parser as BTSDB
```
After that, you only need to include the alias of the parser in the DATASETS constant, in our case, we would imagine that we are only using the BTSDB parser, so our DATASETS constant would be:
```python
DATASETS = [BTSDB] 
```

In case of having multiple dataset parsers, we would only need to import them individually and include them in the DATASETS constant. Example with 4 datasets parsers I used for SaferAuto:
```python
DATASETS = [GTSDB, BTSDB, LISATS, MASTIF] 
```

### 3º Run the general parser
Now you only need to run the general parser. You can do that with:
```
python3 general_parser.py
```

## Author
[**Ángel Igareta**](https://github.com/angeligareta) - Computer Engineering Student

## License
This project is licensed under the **[MIT License](LICENSE)**
