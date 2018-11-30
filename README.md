How to run our trained Faster RCNN on the Road Damage Detection Challenge (RDDC) dataset:

* Copy the dataset into a folder /path/to/dataset which contains the images and annotations of all cities combined:
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

* Get the code:
```bash
git clone https://github.com/fkluger/road_damage_2018.git
cd road_damage_2018
```
* Install required packages (better do this in a separate pip or conda environment):

```bash
pip install -r requirements.txt
export PYTHONPATH=./
```
* Compile custom layers etc.: 

```bash
cd lib
bash make.sh 
cd ../
cd data
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make 
cd ../../..
```
* Create link to dataset:

```bash
ln -s  /path/to/dataset data/rddc_2018
```

* Save the pretrained model from [this Google Drive](https://drive.google.com/open?id=1PDQEXZv5LNuiswFVrJwpD2snGFFjccnk)
into the `checkpoints` folder.

* Compute detections on the test set, replace [GPU_ID] with the ID of the GPU to use (or omit it for CPU mode):
```bash
./experiments/scripts/test_faster_rcnn_for_rddc.sh [GPU_ID]
```

* The detections are saved in the folder `data/rddc_2018/results`, in a file that ends with `det_test_rddc_result.txt`. 
(Each run creates a new file.)

---

This project is based on:
[pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn) by ruotianluo