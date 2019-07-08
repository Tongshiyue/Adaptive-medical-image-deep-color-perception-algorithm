# -*- coding:utf-8 -*-

from __future__ import division, print_function
import numpy as np
import tensorflow as tf
from vgg19.vgg import Vgg19
from PIL import Image
import time
from functools import partial
import copy
import os
from style_swap import *
import matlab.engine
from closed_form_matting import getLaplacian
import imagehash
import argparse
import shelve
import csv
import pickle

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3 6

VGG_MEAN = [103.939, 116.779, 123.68]
Image_Weight = 112
Image_Hight = 112

def rgb2bgr(rgb, vgg_mean=True):
    if vgg_mean:
        return rgb[:, :, ::-1] - VGG_MEAN
    else:
        return rgb[:, :, ::-1]

def bgr2rgb(bgr, vgg_mean=False):
    if vgg_mean:
        return bgr[:, :, ::-1] + VGG_MEAN
    else:
        return bgr[:, :, ::-1]

def load_seg(content_seg_path, style_seg_path, content_shape, style_shape):
    color_codes = ['BLUE', 'GREEN', 'BLACK', 'WHITE', 'RED', 'YELLOW', 'GREY', 'LIGHT_BLUE', 'PURPLE']
    def _extract_mask(seg, color_str):
        h, w, c = np.shape(seg)
        if color_str == "BLUE":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "GREEN":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "BLACK":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "WHITE":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "RED":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "YELLOW":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "GREY":
            mask_r = np.multiply((seg[:, :, 0] > 0.4).astype(np.uint8),
                                 (seg[:, :, 0] < 0.6).astype(np.uint8))
            mask_g = np.multiply((seg[:, :, 1] > 0.4).astype(np.uint8),
                                 (seg[:, :, 1] < 0.6).astype(np.uint8))
            mask_b = np.multiply((seg[:, :, 2] > 0.4).astype(np.uint8),
                                 (seg[:, :, 2] < 0.6).astype(np.uint8))
        elif color_str == "LIGHT_BLUE":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "PURPLE":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        return np.multiply(np.multiply(mask_r, mask_g), mask_b).astype(np.float32)

    # PIL resize has different order of np.shape
    content_seg = np.array(Image.open(content_seg_path).convert("RGB").resize(content_shape, resample=Image.BILINEAR), dtype=np.float32) / 255.0
    style_seg = np.array(Image.open(style_seg_path).convert("RGB").resize(style_shape, resample=Image.BILINEAR), dtype=np.float32) / 255.0

    color_content_masks = []
    color_style_masks = []
    for i in xrange(len(color_codes)):
        color_content_masks.append(tf.expand_dims(tf.expand_dims(tf.constant(_extract_mask(content_seg, color_codes[i])), 0), -1))
        color_style_masks.append(tf.expand_dims(tf.expand_dims(tf.constant(_extract_mask(style_seg, color_codes[i])), 0), -1))

    return color_content_masks, color_style_masks

def gram_matrix(activations):
    height = tf.shape(activations)[1]
    width = tf.shape(activations)[2]
    num_channels = tf.shape(activations)[3]
    gram_matrix = tf.transpose(activations, [0, 3, 1, 2])
    gram_matrix = tf.reshape(gram_matrix, [num_channels, width * height])
    gram_matrix = tf.matmul(gram_matrix, gram_matrix, transpose_b=True)
    return gram_matrix

def content_loss(const_layer, var_layer, weight):
    # const_max = tf.reduce_max(const_layer)
    # var_max = tf.reduce_max(var_layer)                  /const_max         /var_max
    return 0.5 * tf.reduce_mean(tf.squared_difference(const_layer, var_layer)) * weight

def style_loss(CNN_structure, const_layers, var_layers, content_segs, style_segs, weight):
    loss_styles = []
    layer_count = float(len(const_layers))
    layer_index = 0

    _, content_seg_height, content_seg_width, _ = content_segs[0].get_shape().as_list()
    _, style_seg_height, style_seg_width, _ = style_segs[0].get_shape().as_list()
    for layer_name in CNN_structure:
        layer_name = layer_name[layer_name.find("/") + 1:]

        # downsampling segmentation
        if "pool" in layer_name:
            content_seg_width, content_seg_height = int(math.ceil(content_seg_width / 2)), int(math.ceil(content_seg_height / 2))
            style_seg_width, style_seg_height = int(math.ceil(style_seg_width / 2)), int(math.ceil(style_seg_height / 2))

            for i in xrange(len(content_segs)):
                content_segs[i] = tf.image.resize_bilinear(content_segs[i], tf.constant((content_seg_height, content_seg_width)))
                style_segs[i] = tf.image.resize_bilinear(style_segs[i], tf.constant((style_seg_height, style_seg_width)))

        elif "conv" in layer_name:
            for i in xrange(len(content_segs)):
                # have some differences on border with torch
                content_segs[i] = tf.nn.avg_pool(tf.pad(content_segs[i], [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), \
                ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')
                style_segs[i] = tf.nn.avg_pool(tf.pad(style_segs[i], [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), \
                ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')

        if layer_name == var_layers[layer_index].name[var_layers[layer_index].name.find("/") + 1:]:
            print("Setting up style layer: <{}>".format(layer_name))
            const_layer = const_layers[layer_index]
            var_layer = var_layers[layer_index]

            layer_index = layer_index + 1

            layer_style_loss = 0.0
            for content_seg, style_seg in zip(content_segs, style_segs):
                gram_matrix_const = gram_matrix(tf.multiply(const_layer, style_seg))
                style_mask_mean   = tf.reduce_mean(style_seg)
                gram_matrix_const = tf.cond(tf.greater(style_mask_mean, 0.),
                                        lambda: gram_matrix_const / (tf.to_float(tf.size(const_layer)) * style_mask_mean),
                                        lambda: gram_matrix_const
                                    )

                gram_matrix_var   = gram_matrix(tf.multiply(var_layer, content_seg))
                content_mask_mean = tf.reduce_mean(content_seg)
                gram_matrix_var   = tf.cond(tf.greater(content_mask_mean, 0.),
                                        lambda: gram_matrix_var / (tf.to_float(tf.size(var_layer)) * content_mask_mean),
                                        lambda: gram_matrix_var
                                    )

                diff_style_sum    =0.5 * tf.reduce_mean(tf.squared_difference(gram_matrix_const, gram_matrix_var)) * content_mask_mean

                layer_style_loss += diff_style_sum

            loss_styles.append(layer_style_loss * weight)
    return loss_styles

def total_variation_loss(output, weight):
    shape = output.get_shape()
    tv_loss = tf.reduce_sum((output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) * (output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) + \
              (output[:, :-1, :-1, :] - output[:, 1:, :-1, :]) * (output[:, :-1, :-1, :] - output[:, 1:, :-1, :])) / 2.0
    return tv_loss * weight

def save_result(img_, str_):
    result = Image.fromarray(np.uint8(np.clip(img_, 0, 255.0)))
    # output = result.convert("YCbCr")
    result.save(str_)

def affine_loss(output, M, weight):
    loss_affine = 0.0
    output_t = tf.image.rgb_to_grayscale(output) / 255.
    for Vc in tf.unstack(output_t, axis=-1):
        Vc_ravel = tf.reshape(tf.transpose(Vc), [-1])
        loss_affine += tf.matmul(tf.expand_dims(Vc_ravel, 0), tf.sparse_tensor_dense_matmul(M, tf.expand_dims(Vc_ravel, -1)))

    return loss_affine * weight

iter_count = 0

min_loss, best_image = float("inf"), None

def print_loss(args,loss_content, loss_styles_list,loss_tv,loss_affine,styleswap_loss,overall_loss, output_image,loss_style):
    global iter_count, min_loss, best_image, loss_content1, loss_yloss,loss_styleloss,loss_styleswaploss,loss_overallloss

    if iter_count % args.print_iter == 0:
        print('Iteration {} / {}\n\tContent loss: {}'.format(iter_count, args.max_iter, loss_content))
        for j, style_loss in enumerate(loss_styles_list):
            print('\tStyle {} loss: {}'.format(j + 1, style_loss))
        print('\tTV loss: {}'.format(loss_tv))
        print('\tAffine loss: {}'.format(loss_affine))
        print('\tStyleswap loss: {}'.format(styleswap_loss))
        print('\tTotal loss: {}'.format(overall_loss))
        print('\tstyle loss: {}'.format(loss_style))


    if overall_loss < min_loss:
        min_loss, best_image = overall_loss, output_image

    if iter_count % args.save_iter == 0 and iter_count != 0:
        save_result(best_image[:, :, ::-1], os.path.join(args.serial, 'out_iter_{}_l1_swap.png'.format(iter_count)))

    iter_count += 1

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

def y_loss(content_image,output_image,weight=1):
    content_image = np.expand_dims(content_image,0)
    content_image = rgb2yuv(content_image)
    content_image = tf.squeeze(content_image, [0])
    content_h = content_image[:,:,0]

    output_image = rgb2yuv(tf.expand_dims(output_image,0))
    output_image = tf.squeeze(output_image, [0])
    output_h = output_image[:,:,0]

    return 0.5 * tf.reduce_mean(tf.squared_difference(content_h, output_h)) * weight , output_h,content_h
    # return (0.5*tf.reduce_mean(tf.norm(tf.subtract(content_h,output_h),ord=1))+0.5* tf.reduce_mean(tf.squared_difference(content_h, output_h))) * weight, output_h,content_h
    # return tf.reduce_mean(tf.norm(tf.subtract(content_h,output_h),ord=1)) * weight, output_h,content_h

def search(dataset,db_shelve,query):
    h_list = []
    delth = []
    filenames = []
    db = shelve.open(db_shelve)
    with open('h.csv', 'r') as csv_file:
        h_file = csv.reader(csv_file)
        for h in h_file:
            h_list.append(h)
        h_list = np.squeeze(h_list)
    print(type(h_list), h_list.shape)

    query = Image.open(query)
    h_s = str(imagehash.dhash(query))
    print(h_s)
    for h in h_list:
        print(type(h), h)
        difference = int(h, 16) ^ int(h_s, 16)
        delth.append(bin(difference).count("1"))
    delth = np.array(delth)
    delth = np.argsort(delth)
    filenames = h_list[delth[0:1]]
    # filenames = db[filenames]
    print("Found %d images" % (len(filenames)))

    # loop over the images
    for filename in filenames:
        print(filename, type(filename))
        filename = (db[filename])[0]
        print(filename, type(filename))
        image = Image.open(dataset + "/" + filename)
    # close the shelve database
    db.close()
    return os.path.join(dataset + "/" + filename)

def stylize(args, Matting):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    start = time.time()
    # prepare input images
    content_image = Image.open(args.content_image_path).convert("RGB")
    content_image = content_image.resize([Image_Hight,Image_Weight],Image.ANTIALIAS)
    content_image = np.array(content_image, dtype=np.float32)
    content_width, content_height = content_image.shape[1], content_image.shape[0]
    content_image1 = content_image


    content_image = rgb2bgr(content_image)
    content_image = content_image.reshape((1, content_height, content_width, 3)).astype(np.float32)

    args.style_image_path = search(args.images_dataset, args.db_shelve, args.content_image_path)

    style_image = Image.open(args.style_image_path).convert("RGB")
    style_image = style_image.resize([Image_Hight,Image_Weight],Image.ANTIALIAS)
    style_image = rgb2bgr(np.array(style_image, dtype=np.float32))
    style_width, style_height = style_image.shape[1], style_image.shape[0]
    style_image = style_image.reshape((1, style_height, style_width, 3)).astype(np.float32)

    # prepare segment images
    eng = matlab.engine.start_matlab()
    args.content_seg_path = eng.k_means_seg_image(args.content_image_path)
    args.style_seg_path = eng.k_means_seg_image(args.style_image_path)


    content_masks, style_masks = load_seg(args.content_seg_path, args.style_seg_path, [content_width, content_height], [style_width, style_height])
    # os.remove(args.style_seg_path)
    # os.remove(args.content_seg_path)


    if not args.init_image_path:
        if Matting:
            print("<WARNING>: Apply Matting with random init")
        init_image = np.random.randn(1, content_height, content_width, 3).astype(np.float32) * 0.0001
    else:
        init_image = np.expand_dims(rgb2bgr(np.array(Image.open(args.init_image_path).convert("RGB").resize([Image_Hight,Image_Weight],Image.ANTIALIAS), dtype=np.float32)).astype(np.float32), 0)

    mean_pixel = tf.constant(VGG_MEAN)
    input_image = tf.Variable(init_image)

    with tf.name_scope("constant"):
        vgg_const = Vgg19()
        vgg_const.build(tf.constant(content_image), clear_data=False)

        content_fv = sess.run([vgg_const.conv4_2,vgg_const.conv3_1,vgg_const.conv1_1,vgg_const.conv2_1,vgg_const.conv5_1,vgg_const.conv4_1])
        content_layer_const = tf.constant(content_fv[0])
        content_conv3_1_const = content_fv[1]
        content_conv1_1_const = content_fv[2]
        content_conv2_1_const = content_fv[3]
        content_conv4_1_const = content_fv[5]
        content_conv5_1_const = content_fv[4]

        vgg_const.build(tf.constant(style_image))
        style_layers_const = [vgg_const.conv1_1, vgg_const.conv2_1, vgg_const.conv3_1, vgg_const.conv4_1, vgg_const.conv5_1]
        style_fvs = sess.run(style_layers_const)
        style_conv1_1_const = style_fvs[0]
        style_conv2_1_const = style_fvs[1]
        style_conv3_1_const = style_fvs[2]
        style_conv4_1_const = style_fvs[3]
        style_conv5_1_const = style_fvs[4]
        style_layers_const = [tf.constant(fv) for fv in style_fvs]



    with tf.name_scope("variable"):
        vgg_var = Vgg19()
        vgg_var.build(input_image)

    # which layers we want to use?
    style_layers_var = [vgg_var.conv1_1, vgg_var.conv2_1, vgg_var.conv3_1, vgg_var.conv4_1, vgg_var.conv5_1]
    content_layer_var = vgg_var.conv4_2
    content_conv1_1_var = style_layers_var[0]
    content_conv2_1_var = style_layers_var[1]
    content_conv3_1_var = style_layers_var[2]
    content_conv4_1_var = style_layers_var[3]
    content_conv5_1_var = style_layers_var[4]

    # The whole CNN structure to downsample mask
    layer_structure_all = [layer.name for layer in vgg_var.get_all_layers()]

    # Content Loss
    loss_content = content_loss(content_layer_const, content_layer_var, float(args.content_weight))

    # Style Loss
    loss_styles_list = style_loss(layer_structure_all, style_layers_const, style_layers_var, content_masks, style_masks, float(args.style_weight))
    loss_style = 0.0
    for loss in loss_styles_list:
        loss_style += loss

    input_image_plus = tf.squeeze(input_image + mean_pixel, [0])

    #styleswap_loss

    styleswap_loss = 0.0
  #   styleswap = style_swap(content_conv1_1_const, style_conv1_1_const)
  #   styleswap = sess.run(styleswap)
  #   styleswap_loss += tf.reduce_mean(tf.squared_difference(styleswap, content_conv1_1_var)) * args.swapweight

  #   styleswap = style_swap(content_conv2_1_const, style_conv2_1_const)
  #   styleswap = sess.run(styleswap)
  #   #styleswap = tf.constant(styleswap)
  #   styleswap_loss += tf.reduce_mean(tf.squared_difference(styleswap, content_conv2_1_var)) * args.swapweight
  #
    styleswap = style_swap(content_conv3_1_const, style_conv3_1_const)
    styleswap = sess.run(styleswap)
    styleswap_loss += tf.reduce_mean(tf.squared_difference(styleswap, content_conv3_1_var)) * args.swapweight

    styleswap = style_swap(content_conv4_1_const, style_conv4_1_const)
    styleswap = sess.run(styleswap)
    styleswap_loss += tf.reduce_mean(tf.squared_difference(styleswap, content_conv4_1_var)) * args.swapweight

    styleswap = style_swap(content_conv5_1_const, style_conv5_1_const)
    styleswap = sess.run(styleswap)
    styleswap_loss += tf.reduce_mean(tf.squared_difference(styleswap, content_conv5_1_var)) * args.swapweight

    # Affine Loss
    if Matting:
        loss_affine,output_image,content_h= y_loss(content_image1,input_image_plus,args.affine_weight)
    else:
        loss_affine = tf.constant(0.00001)  # junk value

    # Total Variational Loss
    loss_tv = total_variation_loss(input_image, float(args.tv_weight))

    if args.lbfgs:
        if not Matting:
            overall_loss = loss_content + loss_tv + loss_style
        else:
            overall_loss = loss_content + loss_tv + loss_affine + loss_style+ styleswap_loss

        optimizer = tf.contrib.opt.ScipyOptimizerInterface(overall_loss, method='L-BFGS-B', options={'maxiter': args.max_iter, 'disp': 0})
        sess.run(tf.global_variables_initializer())

        print_loss_partial = partial(print_loss, args)
        optimizer.minimize(sess, fetches=[ loss_content, loss_styles_list, loss_tv, loss_affine,styleswap_loss, overall_loss, input_image_plus,loss_style], loss_callback=print_loss_partial)
        global min_loss, best_image, iter_count,loss_content1, loss_yloss,loss_styleloss,loss_styleswaploss,loss_overallloss
        best_result = copy.deepcopy(best_image)
        min_loss, best_image = float("inf"), None
        iter_count = 0
       # end = time.time()
        sess.close()
        return best_result
    else:
        VGGNetLoss = loss_content + loss_tv + loss_style
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
        VGG_grads = optimizer.compute_gradients(VGGNetLoss, [input_image])

        if Matting:
            b, g, r = tf.unstack(input_image_plus / 255., axis=-1)
            b_gradient = tf.transpose(tf.reshape(2 * tf.sparse_tensor_dense_matmul(M, tf.expand_dims(tf.reshape(tf.transpose(b), [-1]), -1)), [content_width, content_height]))
            g_gradient = tf.transpose(tf.reshape(2 * tf.sparse_tensor_dense_matmul(M, tf.expand_dims(tf.reshape(tf.transpose(g), [-1]), -1)), [content_width, content_height]))
            r_gradient = tf.transpose(tf.reshape(2 * tf.sparse_tensor_dense_matmul(M, tf.expand_dims(tf.reshape(tf.transpose(r), [-1]), -1)), [content_width, content_height]))

            Matting_grad = tf.expand_dims(tf.stack([b_gradient, g_gradient, r_gradient], axis=-1), 0) / 255. * args.affine_weight
            VGGMatting_grad = [(VGG_grad[0] + Matting_grad, VGG_grad[1]) for VGG_grad in VGG_grads]

            train_op = optimizer.apply_gradients(VGGMatting_grad)
        else:
            train_op = optimizer.apply_gradients(VGG_grads)

        sess.run(tf.global_variables_initializer())
        min_loss, best_image = float("inf"), None
        for i in xrange(1, args.max_iter):
            _, loss_content_, loss_styles_list_, loss_tv_, loss_affine_, overall_loss_, output_image_ = sess.run([
                train_op, loss_content, loss_styles_list, loss_tv, loss_affine, VGGNetLoss, input_image_plus
            ])
            if i % args.print_iter == 0:
                print('Iteration {} / {}\n\tContent loss: {}'.format(i, args.max_iter, loss_content_))
                for j, style_loss_ in enumerate(loss_styles_list_):
                    print('\tStyle {} loss: {}'.format(j + 1, style_loss_))
                print('\tTV loss: {}'.format(loss_tv_))
                if Matting:
                    print('\tAffine loss: {}'.format(loss_affine_))
                print('\tTotal loss: {}'.format(overall_loss_ - loss_tv_))

            if overall_loss_ < min_loss:
                min_loss, best_image = overall_loss_, output_image_

            if i % args.save_iter == 0 and i != 0:
                save_result(best_image[:, :, ::-1], os.path.join(args.serial, 'out_iter_{}_1.png'.format(i)))

        return best_image

if __name__ == "__main__":
    stylize()
