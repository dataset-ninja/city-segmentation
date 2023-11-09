**CitySegmentation** is a dataset for instance segmentation, semantic segmentation, and object detection tasks. Possible applications of the dataset could be in the geospatial domain. 

The dataset consists of 50 images with 9306 labeled objects belonging to 2 different classes including *building* and *road*.

Images in the CitySegmentation dataset have pixel-level instance segmentation annotations. Due to the nature of the instance segmentation task, it can be automatically transformed into a semantic segmentation (only one mask for every class) or object detection (bounding boxes for every object) tasks. All images are labeled (i.e. with annotations). There are no pre-defined <i>train/val/test</i> splits in the dataset. Additionally, images have ***city*** tag: Berlin or London. The dataset was released in 2019.

<img src="https://github.com/dataset-ninja/city-segmentation/raw/main/visualizations/poster.png">
