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

# content = Image.open("./4/20.jpg").convert("RGB")
# content = content.convert("YCbCr")
# Y, _, _ = content.split()
# style = Image.open("./4/28.jpg").convert("RGB")
# content_seg = Image.open("./4/20_seg2.jpg").convert("RGB")
# style_seg = Image.open("./4/28_seg.jpg").convert("RGB")
#
# content_y, content_u,content_v = RGB2YUV2SPLIT(content)
# content_seg_y, content_seg_u,content_seg_v = RGB2YUV2SPLIT(content_seg)
# style_y, style_u,style_v = RGB2YUV2SPLIT(style)
# style_seg_y, style_seg_u,style_seg_v = RGB2YUV2SPLIT(style_seg)
#
# content_y.save("./20content_y.jpg")
# # content_y.save("./content_y.jpg")
# content_seg_v.save("./20content_seg_v.jpg")
# content_seg_u.save("./20content_seg_u.jpg")
# style_v.save("./style_v.jpg")
# style_seg_v.save("./style_seg_v.jpg")
# style_u.save("./style_u.jpg")
# style_seg_u.save("./style_seg_u.jpg")

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

# a = np.array(U_i)
V_i.save("img_vi.jpg")
U_i.save("img_ui.jpg")


# style = Image.merge("YCbCr", [V_i,V_i, V_i])
# style.show()
# style.save("img_v.jpg")

# V_v = V_v.astype(np.uint8)
# V_u = V_u.astype(np.uint8)
# U_u = U_u.astype(np.uint8)
# U_u = Image.fromarray(U_u)
# V_v = Image.fromarray(V_v)
# V_u = Image.fromarray(V_u)


# style1 = Image.merge("YCbCr", [Y_c, V_u, V_v])
# style1.show()
# style1 = Image.merge("YCbCr", [V_u, V_u, V_u])
# style1 = Image.merge("YCbCr", [U_v, U_v, U_v])
# style1 = Image.merge("YCbCr", [U_u, U_u, U_u])

# Y_c.show()
# Y_c1 = np.array(Y_c)
# img_gray_hist = cv2.calcHist([Y_c1], [0], Y_c1, [256], [0, 256])
# plt.plot(img_gray_hist)
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of Pixels')
# plt.show()


#
# # style = Image.merge("YCbCr", [Y, U, V])
# # style.save("style.jpg")
