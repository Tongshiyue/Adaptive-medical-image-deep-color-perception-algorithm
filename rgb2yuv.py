from PIL import Image,ImageEnhance
import numpy as np
import tensorflow as tf

import cv2
from skimage import io,color

def huv_process(img, img_content, k = 1):
    hsv = img.convert("YCbCr")
    hsv_conten = img_content.convert("YCbCr")

    hsv_hsv = hsv.split()
    hsv_u = hsv_hsv[1]
    hsv_v = hsv_hsv[2]
    hsv_u = k * (np.array(hsv_u))
    hsv_v = k * (np.array(hsv_v))

    hsv_u = Image.fromarray(hsv_u)
    hsv_v = Image.fromarray(hsv_v)
    hsv_v.save("img_vi.jpg")
    hsv_u.save("img_ui.jpg")

    hsv_conten_hsv = hsv_conten.split()
    hsv_conten_h = hsv_conten_hsv[0]
    hsv_conten_h.save("img_yc.jpg")

    style_hsv = Image.merge("YCbCr", (hsv_conten_h, hsv_u, hsv_v))
    style_hsv = style_hsv.convert("RGB")
    # style_hsv.save("MRI-0121_huv.jpg")
    return style_hsv


def rgb2yuv(rgb):
	"""
	Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
	"""
	rgb2yuv_filter = tf.constant([[[[0.299, -0.169, 0.499],
                                    [0.587, -0.331, -0.418],
                                    [0.114, 0.499, -0.0813]]]])
	rgb2yuv_bias = tf.constant([0., 0.5, 0.5])
	temp = tf.nn.conv2d(rgb, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
	temp = tf.nn.bias_add(temp, rgb2yuv_bias)
	return temp

if __name__=="__main__":

    k = 1
    img_content = Image.open("21_224.jpg").convert("RGB")
    img_content = np.array(img_content)
    h= np.shape(img_content)
    img_content = Image.fromarray(img_content)

    img = Image.open("best_stylized.png").convert("RGB")
    img = img.resize([h[0],h[1]],Image.ANTIALIAS)
    img.save("img_big.jpg")

    enh_col = ImageEnhance.Color(img)
    result = enh_col.enhance(color)
#
#     huv_process(result,img_content,1)
