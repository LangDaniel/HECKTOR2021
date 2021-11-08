# Head and neck progression free survival prediction

Code for our survival prediction approach to the MICCAI
[HEad and neCK TumOR segmentation and outcome prediction in PET/CT images (HECKTOR) 2021 challenge](
https://www.aicrowd.com/challenges/miccai-2021-hecktor).

The complete code will be published soon

## Conda environments

There are two files provided to generate conda environments which allow you
to run the code.
The environment generated by `prep_env.yml` is needed for preprocessing
and `train_env.yml` for training.  
Run
```
conda env create --file=<FILE.yaml>
```
to generate the `(prep)` and `(train)` enviroments.

## Data preparation

Put the (zipped) challenge data in the `data/` folder and run the `utils/data_prep.sh` file
with the `(prep)` conda environment.
If you are not on Linux, follow the respective steps specified in the file.

## Training

In order to train the model you have to run:
```
python main.py [--gpu <GPU_INDEX>] parameter/par.yml
```
in the `survival/` folder with the `(train)` conda environment.

### Pretrained weights 

The pre-trained weights can be downloaded from
[here](https://syncandshare.lrz.de/getlink/fiKHFuRVVsqcaR5CpvVweYNb/C3D_weights.h5)
and have to be placed in the `data/weights/` folder.
