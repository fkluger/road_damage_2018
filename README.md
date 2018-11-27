How to train the Faster RCNN on the Road Damage Detection Challenge (RDDC) dataset:

Copy the dataset into a folder /path/to/dataset which contains the images and annotations of all cities combined:
```
/path/to/dataset
|  
└───Annotations 
│   │   train_Adachi_00001.xml
|   |   ...
|   |   train_Sumida_00911.xml
|
└───JPEGImages
    |   test_Adachi_00001.jpg
    |   ...
    |   train_Sumida_00911.jpg
```


* activate python environment with pytorch installed and add current directory to the PYTHONPATH, e.g.:

```bash
source activate pytorch
export PYTHONPATH=./
```
* get code and compile some stuff: 

```bash
git clone 
cd pytorch_faster_rcnn/lib
bash make.sh 
cd ../
cd data/coco/PythonAPI
make 
cd ../../..
```
* create links to dataset and pretrained resnet models:

```bash
ln -s  /path/to/dataset data/rddc_2018
```
