import tensorflow as tf
import numpy as np
import math
PREPROCESS_SIZE = 256
def style_swap(content_feature,style_feature):
    cell_size = 3
    # styfea_h,styfea_w =style_feature.get_shape()[1].value,style_feature.get_shape()[2].value
    # a = [cell_size] * (style_feature.shape[1].value // cell_size) + [style_feature.shape[1].value % cell_size].value
    # c = list(a)
    rows = tf.split(style_feature, num_or_size_splits=list(
            [cell_size] * (style_feature.shape[1] // cell_size) + [style_feature.shape[1] % cell_size]), axis=1)[:-1]
    cells = [tf.split(row, num_or_size_splits=list(
            [cell_size] * (style_feature.shape[2] // cell_size) + [style_feature.shape[2] % cell_size]), axis=2)[:-1]
                 for row in rows]
    stacked_cells = [tf.stack(row_cell, axis=4) for row_cell in cells]
    filters = tf.concat(stacked_cells, axis=-1)
    swaped_end = np.zeros(style_feature.shape,dtype=float).astype(np.float32)
    # style_filter = tf.unstack(filters,axis= 4, num= style_amount)
    swaped_end = tf.add(swaped_end, _swap_op(content_feature, filters))
    return swaped_end

def _swap_op(content_feature, style_feature):
    height = tf.shape(content_feature)[1]
    width = tf.shape(content_feature)[2]

    normalized_filters = tf.nn.l2_normalize(style_feature, dim=(0, 1, 2))
    normalized_filters = tf.squeeze(normalized_filters,[0])
    # normalized_filters = tf.reshape(normalized_filters, (3,3,256,324))
    """ change the strides to see difference"""
    similarity = tf.nn.conv2d(content_feature, normalized_filters, strides=[1, 1, 1, 1], padding="SAME")
    arg_max_filter = tf.argmax(similarity, axis=3) #-1
    one_hot_filter = tf.one_hot(arg_max_filter, depth=similarity.get_shape()[-1].value)
    # style_feature = tf.reshape(style_feature,[3,3,256,324])
    style_feature = tf.squeeze(style_feature,[0])
    swap = tf.nn.conv2d_transpose(one_hot_filter, style_feature, output_shape=tf.shape(content_feature),
                                  strides=[1, 1, 1, 1], padding="SAME")#VALID
    return swap/9.0




def swap_loss(CNN_structure, const_content_layers,const_style_layers, var_layers, content_segs, style_segs, weight):
    loss_styles = []
    layer_count = float(len(const_style_layers))
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
            const_style_layer = const_style_layers[layer_index]
            var_layer = var_layers[layer_index]
            const_content_layer = const_content_layers[layer_index]

            layer_index = layer_index + 1

            layer_style_loss = 0.0
            for content_seg, style_seg in zip(content_segs, style_segs):
                const_style_layer_seg = tf.multiply(const_style_layer, style_seg)
                const_content_layer_seg = tf.multiply(const_content_layer, content_seg)
                var_layer_seg =  tf.multiply(var_layer, content_seg)

                swaped_end = style_swap(const_content_layer_seg,const_style_layer_seg)

                diff_style_sum    =0.5 * tf.reduce_mean(tf.squared_difference(swaped_end, var_layer_seg))

                layer_style_loss += diff_style_sum

            loss_styles.append(layer_style_loss * weight)

    return loss_styles

