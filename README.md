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

## Main files 
### Common Config (common_config.py)
All the common methods for the specific dataset parsers are contained in this file, for instance: read_image, resize_image, write_data and so on. Feel free to check them out, each one is documented. 

On top of that, there are several constants that you can change according to your preferences. These are:
- **MAX_WIDTH, MAX_HEIGHT**: Dimensions of the output images.
- **OUTPUT_IMG_EXTENSION**: Extension of the output images. (Default: jpg)
- **COLOR MODE**: Default -1 (RGB). If you want to read the images in black-white scale use 0 option.
- **SHOW_IMG**: If activated it will show each processed image and the annotated bounding boxes in it.
- **TRAIN_PROB, TEST_PROB**: Percentage of train-test proportion for the input images.

### General parser (general_parser.py) 
Main file of the program. It imports all the specific datasets and loop over them calling the read_dataset method that returns the count of the classes read in the specific dataset. 

At the end, it shows the total number of annotated images per class and train-test proportion.

### Datasets Parsers
Directory that contains all the specific datasets parsers. 

## How to add a new dataset
### 1º Add your dataset parser
As each dataset has specific annotation formats, we need a specific parser for each one. However, most of the methods are common, so the process is not that complicated.

The methods the dataset parser must have are:
- *initialize_traffic_sign_classes()*: This method creates the relations between the real class id of an object in the dataset and the class id we are using for that object in the program. For instance, if a yield object has the class id "A02", we would need to add a relation such as:
```python 
def initialize_traffic_sign_classes():
    traffic_sign_classes["4-yield"] = [6]
```
-*calculate_darknet_format (input_img, image_width, image_height, row)*: This method converts the specific annotation format for a dataset to the darknet format. First of all we calculate the width and height proportion for the image. After that, we need to retrieve the bounding boxes borders from the specific dataset row and calculate the new positions according to the width and height proportions. Finallu we use the parse_darknet_format common method that needs the left_x, bottom_y, right_x and top_y values. 
```python
def calculate_darknet_format(input_img, image_width, image_height, row):
    real_img_width, real_img_height = get_img_dim_plt(input_img)
    width_proportion = (real_img_width / MAX_WIDTH)
    height_proportion = (real_img_height / MAX_HEIGHT)

    left_x = float(row[1]) / width_proportion
    bottom_y = float(row[2]) / height_proportion
    right_x = float(row[3]) / width_proportion
    top_y = float(row[4]) / height_proportion

    object_class = int(row[6])
    object_class_adjusted = adjust_object_class(object_class)  # Adjust class category

    if SHOW_IMG:
        show_img(resize_img_plt(input_img), left_x, bottom_y, (right_x - left_x), (top_y - bottom_y))

    return parse_darknet_format(object_class_adjusted, image_width, image_height, left_x, bottom_y, right_x, top_y)
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