


import argparse
from PIL import Image
from PIL import ImageEnhance
import csv
import numpy as np
import pickle
import os
from photo_style_aaa import stylize
import matplotlib.pyplot as plt
import imagehash
import shelve
import glob2
from rgb2yuv import huv_process
from skimage import color



#os.environ["CUDA_VISIBLE_DEVICES"]="4"
parser = argparse.ArgumentParser()
# Input Options
parser.add_argument("--content_image_path", dest='content_image_path',  nargs='?',
                    help="Path to the content image",default="")#0_224.jpgMRI-014.jpg
parser.add_argument("--style_image_path",   dest='style_image_path',    nargs='?',
                    help="Path to the style image",default="")#./4/28.jpg
parser.add_argument("--content_seg_path",   dest='content_seg_path',    nargs='?',
                    help="Path to the style segmentation",default="")
parser.add_argument("--style_seg_path",     dest='style_seg_path',      nargs='?',
                    help="Path to the style segmentation",default="")
parser.add_argument("--init_image_path",    dest='init_image_path',     nargs='?',
                    help="Path to init image", default="")#./4/normal.jpg
parser.add_argument("--output_image",       dest='output_image',        nargs='?',
                    help='Path to output the stylized image', default="")
parser.add_argument("--output_image2",       dest='output_image2',        nargs='?',
                    help='Path to output the stylized image', default="./results/1/best_stylized2.png")
parser.add_argument("--serial",             dest='serial',              nargs='?',
                    help='Path to save the serial out_iter_X.png', default='./results/1')
parser.add_argument("--images_dataset",       dest='images_dataset',        nargs='?',
                    help='Path to output the reference images', default="./images")
parser.add_argument("--db_shelve",       dest='db_shelve',        nargs='?',
                    help='Path to output the db_shelve', default="./db.shelve")

# Training Optimizer Options
parser.add_argument("--max_iter",           dest='max_iter',            nargs='?', type=int,
                    help='maximum image iteration', default=1000)
parser.add_argument("--learning_rate",      dest='learning_rate',       nargs='?', type=float,
                    help='learning rate for adam optimizer', default=1.0)
parser.add_argument("--print_iter",         dest='print_iter',          nargs='?', type=int,
                    help='print loss per iterations', default=1)
# Note the result might not be smooth enough since not applying smooth for temp result
parser.add_argument("--save_iter",          dest='save_iter',           nargs='?', type=int,
                    help='save temporary result per iterations', default=100)
parser.add_argument("--lbfgs",              dest='lbfgs',               nargs='?',
                    help="True=lbfgs, False=Adam", default=True)

# Weight Options
parser.add_argument("--content_weight",     dest='content_weight',      nargs='?', type=float,
                    help="weight of content loss", default=1e0)
parser.add_argument("--style_weight",       dest='style_weight',        nargs='?', type=float,
                    help="weight of style loss", default=1e2)#1e2
parser.add_argument("--tv_weight",          dest='tv_weight',           nargs='?', type=float,
                    help="weight of total variational loss", default=1e-3)
parser.add_argument("--affine_weight",      dest='affine_weight',       nargs='?', type=float,
                    help="weight of affine loss", default=1e5)#1e4
parser.add_argument("--swapweight",      dest='swapweight',       nargs='?', type=float,
                    help="weight of swap loss", default=1e1)#1e1

# Style Options
parser.add_argument("--style_option",       dest='style_option',        nargs='?', type=int,
                    help="0=non-Matting, 1=only Matting, 2=first non-Matting, then Matting", default=2)
parser.add_argument("--apply_smooth",       dest='apply_smooth',        nargs='?',
                    help="if apply local affine smooth", default=True)#False


# Smoothing Argument
parser.add_argument("--f_radius",           dest='f_radius',            nargs='?', type=int,
                    help="smooth argument", default=15)
parser.add_argument("--f_edge",             dest='f_edge',              nargs='?', type=float,
                    help="smooth argument", default=1e-1)

args = parser.parse_args()

def plaint_line(loss_contents):
    l = len(loss_contents)
    n = 50
    batch_line = l/n
    loss = []
    for i in range(0,l,batch_line):
        a = np.mean(loss_contents[i:i+batch_line])
        loss.append(a)
    return loss

def index(imagespath,db_shelve):
    hlist = []
    db = shelve.open(db_shelve, writeback=True)
    for imagePath in glob2.glob(imagespath + "/*.jpg"):
        # load the image and compute the difference hash
        image = Image.open(imagePath)
        image = image.convert('L')
        # image.show()
        h = str(imagehash.dhash(image))
        print(h)

        filename = imagePath[imagePath.rfind("/") + 1:]
        del db[h]
        db[h] = db.get(h, []) + [filename]
        hlist.append(h)
        print (h, db[h])

    with open('h.csv', 'w') as read_file:
        writer = csv.writer(read_file)
        writer.writerow(hlist)
    # close the shelf database
    db.close()

def main():
    Image_high = Image_Weigh = 224
    if args.style_option == 0:
        index(args.images_dataset,args.db_shelve)
        args.max_iter = 1 * args.max_iter
        best_image_bgr,loss_swap,loss_yloss = stylize(args, False)

        # plant content_loss
        loss = plaint_line(loss_swap)
        loss1 = plaint_line(loss_yloss)
        losses = [loss,loss1]
        with open('swap_yloss_loss.pkl', 'w')as f:
            pickle.dump(losses, f)

        fig, ax1 = plt.subplots()
        lns1 = ax1.plot(np.arange(len(loss)),loss,label = "content_loss")
        ax1.set_xlabel("iter")
        ax1.set_ylabel("content_loss")
        plt.legend(lns1,loc=0)
        plt.savefig("content_loss.png", format='png')
        plt.show()

        result = Image.fromarray(np.uint8(np.clip(best_image_bgr[:, :, ::-1], 0, 255.0)))
        result.save(args.output_image)

    elif args.style_option == 1:
        index(args.images_dataset, args.db_shelve)
        args.max_iter = 0.01 * args.max_iter
        best_image_bgr,loss_contents = stylize(args, True)

        if not args.apply_smooth:
            result = Image.fromarray(np.uint8(np.clip(best_image_bgr[:, :, ::-1], 0, 255.0)))
            filename = args.content_image_path[args.content_image_path.rfind("/") + 1:]
            result.save(os.path.join(args.output_image, filename))
        else:
            # Pycuda runtime incompatible with Tensorflow
            from smooth_local_affine import smooth_local_affine
            content_input = Image.open(args.content_image_path).convert("RGB")
            content_input = content_input.resize([Image_high,Image_Weigh],Image.ANTIALIAS)
            content_input = np.array(content_input, dtype=np.float32)
            # RGB to BGR
            content_input = content_input[:, :, ::-1]
            # H * W * C to C * H * W
            content_input = content_input.transpose((2, 0, 1))
            input_ = np.ascontiguousarray(content_input, dtype=np.float32) / 255.

            _, H, W = np.shape(input_)

            output_ = np.ascontiguousarray(best_image_bgr.transpose((2, 0, 1)), dtype=np.float32) / 255.
            best_ = smooth_local_affine(output_, input_, 1e-7, 3, H, W, args.f_radius, args.f_edge).transpose(1, 2, 0)
            result = Image.fromarray(np.uint8(np.clip(best_ * 255., 0, 255.)))
            enh_col = ImageEnhance.Color(result)
            color = 1.5
            result = enh_col.enhance(color)
            # result.show()
            # image_colored.save("image_colored.jpg")
            filename = args.content_image_path[args.content_image_path.rfind("/") + 1:]
            result.save(os.path.join(args.output_image, filename))

    elif args.style_option == 2:
        args.init_image_path = None
        index(args.images_dataset, args.db_shelve)
        args.max_iter = 2*1000
        tmp_image_bgr, loss_contents = stylize(args, False)
        # loss = plaint_line(loss_contents)
        result = Image.fromarray(np.uint8(np.clip(tmp_image_bgr[:, :, ::-1], 0, 255.0)))
        args.init_image_path = os.path.join(args.serial, "tmp_result.png")
        result.save(args.init_image_path)
        args.max_iter = 1000
        best_image_bgr, loss_contents1 = stylize(args, True)


        if not args.apply_smooth:
            result = Image.fromarray(np.uint8(np.clip(best_image_bgr[:, :, ::-1], 0, 255.0)))
            enh_col = ImageEnhance.Color(result)
            color = 1.5
            result = enh_col.enhance(color)
            filename = args.content_image_path[args.content_image_path.rfind("/") + 1:]
            result.save(os.path.join(args.output_image, filename))
        else:
            from smooth_local_affine import smooth_local_affine
            # content_input = np.array(Image.open(args.content_image_path).convert("RGB"), dtype=np.float32)
            content_input1 = Image.open(args.content_image_path).convert("RGB")
            content_input = content_input1.resize([Image_high,Image_Weigh],Image.ANTIALIAS)
            content_input = np.array(content_input, dtype=np.float32)
            # RGB to BGR
            content_input = content_input[:, :, ::-1]
            # H * W * C to C * H * W
            content_input = content_input.transpose((2, 0, 1))
            input_ = np.ascontiguousarray(content_input, dtype=np.float32) / 255.

            _, H, W = np.shape(input_)

            output_ = np.ascontiguousarray(best_image_bgr.transpose((2, 0, 1)), dtype=np.float32) / 255.
            best_ = smooth_local_affine(output_, input_, 1e-7, 3, H, W, args.f_radius, args.f_edge).transpose(1, 2, 0)
            result = Image.fromarray(np.uint8(np.clip(best_ * 255., 0, 255.)))

            # img_content = content_input1
            # img_content = np.array(img_content)
            # h = np.shape(img_content)
            # print(h)
            # img_content = Image.fromarray(img_content)

            # img = result.resize([h[1], h[0]], Image.ANTIALIAS)
            # img = np.array(img)
            # h2 = np.shape(img)
            # img = Image.fromarray(img)
            # print(h2)
            # # enh_col = ImageEnhance.Color(img)
            # # color = 1.5
            # # result = enh_col.enhance(color)
            # result = huv_process(img, img_content, 1)

            filename = args.content_image_path[args.content_image_path.rfind("/") + 1:]
            result.save(os.path.join(args.output_image,filename))

if __name__ == "__main__":
    path = './datasets'
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            for dirpath2, _, filenames2 in os.walk(os.path.join(dirpath,dirname)):
                for filename in filenames2:
                    args.content_image_path = os.path.join(dirpath2,filename)
                    args.output_image = os.path.join('./results/dataset_result',dirname)
                    main()
                    os.remove(args.content_image_path)







#
# import numpy as np
# import scipy.sparse as sps
# import math
# import tensorflow as tf
# from PIL import Image
# import pickle
# import matplotlib.pyplot as plt
# # W = [[[1,0,3],[1,2,9],[4,3,6]],
# #      [[3,7,6],[7,4,7],[9,6,0]],
# #      [[1,2,8],[5,1,9],[5,8,7]],
# #      [[1,1,1],[4,4,4],[3,3,3]]]
# # row_inds = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
# # col_inds = [0,1,2,3,1,2,3,0,2,3,0,1,3,0,1,2]
# # d = [[1,0,3,1,2,9,4,3],[7,4,7,9,6,0,1,2]]
# # b = [[2,1,1,2,3,4,5,6],[6,2,7,9,6,0,1,2]]
# # d = [0,9,8,7,6,5,4,3,2,1,1,2,3,4,5,6]
# # b = [1,0,3,1,2,9,4,3,2,1,1,2,3,4,5,6]
# # a = sps.csr_matrix((d, (row_inds, col_inds)), shape=(4, 4))
# #
# # a = [[[1,0,3],[1,2,9],[4,3,6]],
# #      [[3,7,6],[7,4,7],[9,6,0]],
# #      [[1,2,8],[5,1,9],[5,8,7]]]
# # b = [[1,1,1],[4,4,4],[3,3,3]]
# # print(map(lambda x:x/2,b[0]))
# # print(np.shape(a))
# # # content_image = np.array(Image.open("./4/normal224.jpg").convert("RGB"),dtype=np.float32)
# # # style_image = np.array(Image.open("./4/28.jpg").convert("RGB"),dtype=np.float32)
# # # weigth = 1
# # sess = tf.Session()
# # d = tf.constant(d,dtype=np.float32)
# # b = tf.constant(b,dtype=np.float32)
# # c = tf.norm(tf.subtract(d,b),ord=2)
# # a = tf.norm(tf.subtract(d,b),ord=1)
# # f = tf.reduce_mean(tf.squared_difference(d, b))
# # e = tf.squared_difference(d, b)
# # g = tf.subtract(d,b)
# # c = sess.run(c)
# # a = sess.run(a)
# # f = sess.run(f)
# # e = sess.run(e)
# # g = sess.run(g)
# # print (c)
# # print (a)
# # print (f)
# # print (g)
# # print (e)



