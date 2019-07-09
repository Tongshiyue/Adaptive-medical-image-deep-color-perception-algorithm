# Adaptive-medical-image-deep-color-perception-algorithm
"Adaptive medical image deep color perception algorithm"

This is a part of our code, if you need whole code, you can email author. 

# Setup
This code is based on tensorflow. It has been tested on Ubuntu 14.04 LTS.

Dependencies:

* Tensorflow 1.40
* Matlab R2016b
* Imagehash
* Numpy
* Pillow
* PyCUDA  
### It is recommended to use [Anaconda Python](https://www.continuum.io/anaconda-overview), since you only need to install Tensorflow and PyCUDA manually to setup. The CUDA is optional but really recommended

CUDA backend:

* CUDA
* cudnn

## VGG-19 model weights

The VGG-19 model of tensorflow is adopted from [VGG Tensorflow](https://github.com/machrisaa/tensorflow-vgg) with few modifications on the class interface. The VGG-19 model weights is stored as .npy file and could be download from [Google Drive](https://drive.google.com/file/d/0BxvKyd83BJjYY01PYi1XQjB5R0E/view) or [BaiduYun Pan](https://pan.baidu.com/s/1o9weflK). After downloading, you need fine tune the weight of VGG19 by medical images. Then copy the fine-tuned weight file to the ./project/vgg19 directory.

# Usage

## Basic Usage

You need to specift the path of the content image and then run the command

```python
python Adaptive_MIdeepcolor.py --content_image_path <path_to_content_image> --color_option 2
```

Example:

```python
python Adaptive_MIdeepcolor.py --content_image_path ./ct.jpg --color_option 2
```
## Other Options

`--color_option` specifies three different ways of medical image colorization. `--color_option 0` is to generate colored medical image without Y-loss. This result is similary to artistic work, its texture details can't be preserved. `--color_option 1` is to generate colored medical image with Y-loss, its texture details is preserved.
`--color_option 2` is to set the generated result of `--color_option 0` as the initial image of `--color_option 1`, then use Y-loss to constrain texture details.

`--content_weight` specifies the weight of the content loss (default=1), `--color_weight` specifies the weight of the style loss (default=100), `--tv_weight` specifies the weight of variational loss (default=1e-3) and `--Y-loss_weight` specifies the weight of Y-loss loss (default=1e5). You can change the values of these weight and play with them to create different photos. `--swapweight` specifies the weight of swap loss (default=1e1). You can change the values of these weight and play with them to create better photos.

#Examples
Here are more results from our algorithm.


# Acknowledgement

We encode our code build upon [LouieYang/deep-photo-styletransfer-tf](https://github.com/LouieYang/deep-photo-styletransfer-tf)

# Citation

If you find this code useful for your research, please cite:

>@misc{YangPhotoStyle2017,
>  author = {Yang Liu},
>  title = {deep-photo-style-transfer-tf},
>  publisher = {GitHub},
>  organization={Alibaba-Zhejiang University Joint Research Institute of Frontier Technologies},
>  year = {2017},
>  howpublished = {\url{https://github.com/LouieYang/deep-photo-styletransfer-tf}}
>}

and

>@misc{Adaptive medical image deep color perception algorithm2019,
>  author = {Shiyue Tong},
>  title = {Adaptive medical image deep color perception algorithm},
>  publisher = {GitHub},
>  year = {2019},
>  howpublished = {\url{https://github.com/Tongshiyue/Adaptive-medical-image-deep-color-perception-algorithm}}
>}

# Contact

Feel free to contact me if you need the whole code or there is any question (zengxh@cqupt.edu.cn)
