from PIL import Image,ImageEnhance
import numpy as np
import cv2
from matplotlib import pyplot as plt

def RGB2YUV2SPLIT(img):
    seg_yuu = img.convert("YCbCr")
    Y, U, V = seg_yuu.split()
    img_Y = Image.merge("YCbCr", [Y, Y, Y])
    img_Y = img_Y.convert("RGB")
    img_u = Image.merge("YCbCr", [U, U, U])
    img_u = img_u.convert("RGB")
    img_v = Image.merge("YCbCr", [V, V, V])
    img_v = img_v.convert("RGB")
    return img_Y,img_u,img_v


style_c = Image.open("MRI-0121.jpg").convert("YCbCr")
img_content = np.array(style_c)
h = np.shape(img_content)
img_content = Image.fromarray(img_content)
img = Image.open("MRI-012.jpg").convert("YCbCr")
img = img.resize([h[0], h[1]], Image.ANTIALIAS)
# img.save("img_big.jpg")
enh_col = ImageEnhance.Color(img)
color = 1.5
result = enh_col.enhance(color)

Y_c, U_c,V_c = style_c.split()
Y_i, U_i,V_i = result.split()

V_i.save("img_vi.jpg")
U_i.save("img_ui.jpg")


