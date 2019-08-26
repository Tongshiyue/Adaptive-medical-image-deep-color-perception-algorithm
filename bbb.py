import os
# import numpy as np
from PIL import Image
import imagehash
import os
from os.path import join as pjoin
import numpy as np
import pickle
import matlab.engine
import shutil

# # t1= pickle.load(open('./images/time3.pkl'))
# t1= pickle.load(open('./time.pkl'))
# a = t1[1:]
# b = np.mean(a)
# c = [224,b]
#
# with open('time_plant.pkl', 'w')as f:
#     pickle.dump(c, f)
#
# print(a)
# print(np.mean(a))
# #
# path = "/home/tsy/code_now/color transfer/datasets/jiaozhi"
# filename = "21.jpg"
# dirpath2 = "/home/tsy/code_now/color transfer/datasets/jiaozhi_seg"
# for dirpath, dirnames, filenames in os.walk(path):
#    for filename in filenames:
#        b = filename.split(".jpg")
#        seg_path = os.path.join(dirpath2,b[0]+"_seg.jpg")
#        print(seg_path)

# for i in range(10):
#     if os.path.exists("./1111.pkl"):
#         b = pickle.load(open("./1111.pkl"))
#         # losses.append(b)
#         # losses.append(i)
#         b.append(i)
#     else:b= [i]
#     with open('./1111.pkl', 'w')as f:
#         pickle.dump(b, f)
a = pickle.load(open("./images/56.pkl"))
print(a)
b = pickle.load(open("./images/time5.pkl"))
print(b)

         # eng = matlab.engine.start_matlab()
#        content_seg_path = eng.k_means_seg_image(os.path.join(path,filename))
#        shutil.move(content_seg_path,dirpath2)


# path = './datasets/z'
# list = os.listdir(path)
# # name = np.arange(len(list))
# for i,filename in enumerate(list) :
#     file = os.path.join(path,filename)
#     image = Image.open(file)
#     image.save(os.path.join(path,"x{}.png".format(i)))
#     os.remove(file)

# from smooth_local_affine import smooth_local_affine


# def load_data(path):
#     images = []
#     count = 0
#     for i in os.listdir(path):
#         image_dir = pjoin(path, i)
#         image = Image.open(image_dir).convert("RGB")
#         images.append(image)
#         count = count + 1
#
#     return count, images
#
# data_dir = './ccc'
# files = os.listdir(data_dir)
# gen = pjoin(data_dir, files[0])
# raw = pjoin(data_dir, files[1])
# count, images1 = load_data(gen)
# _, images2 = load_data(raw)#colored
#
# for i in range(count):
#     img = images2[i]
#     content_input = images1[i]
#     content_input = content_input.resize([224,224],Image.ANTIALIAS)
# # content_input = np.array(Image.open(args.content_image_path).convert("RGB"), dtype=np.float32)
# #     content_input = Image.open(args.content_image_path).convert("RGB")
#     content_input = np.array(content_input, dtype=np.float32)
#     best_image_bgr = np.array(img)
#     # RGB to BGR
#     content_input = content_input[:, :, ::-1]
#     # H * W * C to C * H * W
#     content_input = content_input.transpose((2, 0, 1))
#     input_ = np.ascontiguousarray(content_input, dtype=np.float32) / 255.
#
#     _, H, W = np.shape(input_)
#
#     output_ = np.ascontiguousarray(best_image_bgr.transpose((2, 0, 1)), dtype=np.float32) / 255.
#     best_ = smooth_local_affine(output_, input_, 1e-7, 3, H, W, 15, 1e-1).transpose(1, 2, 0)
#     result = Image.fromarray(np.uint8(np.clip(best_ * 255., 0, 255.)))
#     result.save("./ccc/{}.jpg".format(i))

# a = [31.0378,36.4431,20.5209,28.003,12.4358,20.4145]
# a1 = [35.58]
# b = [34.0031,37.6160,23.5478,29.202,12.8339,21.3427]
# b1 = [36.179]
# c = [26.4986, 37.1917, 25.6797, 29.264,10.889,15.7854]
# c1 = [42.363]
# a = [0.9328,0.9403,0.7683,0.6370,0.1989,0.4543]
# a = [28.003]
# a1=[35.58]
# # b = [0.9554,0.8487,0.8882,0.7140,0.5676,0.4265]
# b = [29.202]
# b1 = [36.179]
# # c = [0.9673,0.8376,0.8404,0.6910,0.2017,0.1993]
# c =[29.264]
# c1 = [42.363]
# d = []
# # for i in range(len(b)):
# #     d.append(b1[0]-b[i])
# # d = (np.array(d))/b1[0]
# print((b1[0]-b[0])/b1[0])
# print((a1[0]-a[0])/a1[0])
# print((c1[0]-c[0])/c1[0])
# m = ((b1[0]-b[0])/b1[0]+(a1[0]-a[0])/a1[0]+(c1[0]-c[0])/c1[0])/3
# print(m)


# a = [1,2,3]
# query = './MRI-012.jpg'
# query = Image.open(query)
# h_s = str(imagehash.dhash(query))
# print(h_s)
# print(int(h_s, 16))