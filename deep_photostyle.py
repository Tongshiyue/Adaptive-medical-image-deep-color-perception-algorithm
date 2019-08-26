import argparse
from PIL import Image
from PIL import ImageEnhance
import csv
import numpy as np
import pickle
import os
from photo_style import stylize
import matplotlib.pyplot as plt
import imagehash
import shelve
import glob2
from rgb2yuv import huv_process
from skimage import io,color


#os.environ["CUDA_VISIBLE_DEVICES"]="4"
parser = argparse.ArgumentParser()
# Input Options
parser.add_argument("--content_image_path", dest='content_image_path',  nargs='?',
                    help="Path to the content image",default="./21_224.jpg")#0_224.jpgMRI-014.jpg
parser.add_argument("--style_image_path",   dest='style_image_path',    nargs='?',
                    help="Path to the style image",default="./21_2.jpg")#./4/28.jpg
parser.add_argument("--content_seg_path",   dest='content_seg_path',    nargs='?',
                    help="Path to the style segmentation",default="")
parser.add_argument("--style_seg_path",     dest='style_seg_path',      nargs='?',
                    help="Path to the style segmentation",default="")
parser.add_argument("--init_image_path",    dest='init_image_path',     nargs='?',
                    help="Path to init image", default="")#./4/normal.jpg
parser.add_argument("--output_image",       dest='output_image',        nargs='?',
                    help='Path to output the stylized image', default="./results/1/best_stylized.png")

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
                    help="weight of swap loss", default=1e1)#1e3

# Style Options
parser.add_argument("--style_option",       dest='style_option',        nargs='?', type=int,
                    help="0=non-Matting, 1=only Matting, 2=first non-Matting, then Matting", default=1)
parser.add_argument("--apply_smooth",       dest='apply_smooth',        nargs='?',
                    help="if apply local affine smooth", default=True)


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
        # if db[h] !=  None:
        #     del db[h]
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
        best_image_bgr,loss_contents = stylize(args, False)

        # plant content_loss
        loss = plaint_line(loss_contents)
        losses = [loss]
        with open('loss.pkl', 'w')as f:
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
        args.max_iter = 1 * args.max_iter
        args.init_image_path = os.path.join(args.serial, "tmp_result.png")

        # best_image_bgr,loss_contents = stylize(args, True)
        # loss_contents = plaint_line(loss_contents)

        best_image_bgr,loss1,loss2,loss3,loss4,loss5 = stylize(args, True)
        # loss1 = plaint_line(loss1)
        # loss2 = plaint_line(loss2)
        # loss3 = plaint_line(loss3)
        # loss4 = plaint_line(loss4)
        # loss5 = plaint_line(loss5)

        # loss1,loss2,loss3,loss4= pickle.load(open('loss_onetime.pkl'))
        losses = [loss1,loss2,loss3,loss4,loss5]
        with open('loss_onetime3.pkl', 'w')as f:
            pickle.dump(losses, f)
        # plant content_loss
        # loss = plaint_line(loss_contents)
        # loss1,loss2,loss3,loss4 = pickle.load(open('loss.pkl'))
        # losses = [loss1,loss2,loss3,loss4,loss]
        # with open('loss.pkl', 'w')as f:
        #     pickle.dump(losses, f)
        # fig, ax1 = plt.subplots()
        # lns1 = ax1.plot(np.arange(len(loss)),loss,label = "y_loss")
        # ax1.set_xlabel("iter")
        # ax1.set_ylabel("y_loss")
        # plt.legend(lns1,loc=0)
        # plt.savefig("y_loss.png", format='png')
        # plt.show()

        if not args.apply_smooth:
            result = Image.fromarray(np.uint8(np.clip(best_image_bgr[:, :, ::-1], 0, 255.0)))
            result.save(args.output_image)
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
            result.save(args.output_image)
    elif args.style_option == 2:
        index(args.images_dataset, args.db_shelve)
        args.max_iter = 1 * args.max_iter
        tmp_image_bgr, loss_contents = stylize(args, False)
        # loss = plaint_line(loss_contents)
        losses = [loss_contents]
        with open('loss.pkl', 'w')as f:
            pickle.dump(losses, f)

        result = Image.fromarray(np.uint8(np.clip(tmp_image_bgr[:, :, ::-1], 0, 255.0)))
        args.init_image_path = os.path.join(args.serial, "tmp_result.png")
        result.save(args.init_image_path)
        args.max_iter = 1 * args.max_iter
        best_image_bgr, loss_contents1 = stylize(args, True)
        loss1 = pickle.load(open('loss.pkl'))
        losses = [loss1,loss_contents]
        with open('loss.pkl', 'w')as f:
            pickle.dump(losses, f)


        # # plant
        # loss2 = plaint_line(loss_contents1)
        # fig,ax1 = plt.subplots(figsize=(7.5,4.8))
        # ax2 = ax1.twinx()
        # lns1 = ax1.plot(np.arange(len(loss1)),loss1,label = "content_loss",linewidth=1)
        # lns2 = ax2.plot(np.arange(len(loss_contents)), loss_contents, label="y_loss",color="red",linewidth=1)
        # ax1.set_xlabel('iteration')
        # ax1.set_ylabel('content_loss')
        # ax2.set_ylabel('y_loss')
        # lns = lns1 + lns2
        # labels = ["content_loss","y_loss"]
        # plt.legend(lns,labels,loc=7)
        # plt.savefig("content_y_loss_compare.png", format='png')
        # plt.show()

        # loss = plaint_line(loss_contents)
        #
        # fig, ax1 = plt.subplots()
        # lns1 = ax1.plot(np.arange(len(loss)),loss,label = "y_loss")
        # ax1.set_xlabel("iter")
        # ax1.set_ylabel("y_loss")
        # plt.legend(lns1,loc=0)
        # plt.savefig("y_loss.png", format='png')
        # plt.show()

        # losses = [loss, loss2]
        # with open('loss.pkl', 'w')as f:
        #     pickle.dump(losses, f)

        # loss, loss2 = pickle.load(open('loss.pkl'))

        if not args.apply_smooth:
            result = Image.fromarray(np.uint8(np.clip(best_image_bgr[:, :, ::-1], 0, 255.0)))
            result.save(args.output_image)
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

            img_content = content_input1
            img_content = np.array(img_content)
            h = np.shape(img_content)
            img_content = Image.fromarray(np.uint8(img_content))

            img = result.resize([h[1], h[0]], Image.ANTIALIAS)

            result = huv_process(img, img_content, 1)

            result.save(args.output_image)
            # filename = args.content_image_path[args.content_image_path.rfind("/") + 1:]
            # result.save(os.path.join(args.output_image, filename))

if __name__ == "__main__":
    main()
