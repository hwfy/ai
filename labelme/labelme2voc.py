#!/usr/bin/env python

# labelMe格式转VOC2007数据集格式
# 官方示例：https://github.com/wkentaro/labelme/tree/v3.11.2/examples/instance_segmentation

from __future__ import print_function

import argparse
import glob
import json
import os
import os.path as osp
import sys

import numpy as np
import PIL.Image

# pip install labelme==3.16.2
import labelme


def main():
    # 命令行参数解析
    # ArgumentDefaultsHelpFormatter：添加默认的值的信息到每一个帮助信息的参数中
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 打标目录
    parser.add_argument('--input_dir', default='labels', help='input annotated directory')
    # 输出目录
    parser.add_argument('--output_dir', default='output', help='output dataset directory')
    # 打标分类文件
    parser.add_argument('--labels', default='label.txt', help='labels file', required=False)
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print('Output directory already exists:', args.output_dir)
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, 'JPEGImages'))
    os.makedirs(osp.join(args.output_dir, 'SegmentationClass'))
    os.makedirs(osp.join(args.output_dir, 'SegmentationClassPNG'))
    os.makedirs(osp.join(args.output_dir, 'SegmentationClassVisualization'))
    print('Creating dataset:', args.output_dir)

    # 保存分类名
    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        elif class_id == 0:
            assert class_name == '_background_'
        class_names.append(class_name)
    # 转换成元组（小括号），元组的元素不能修改
    class_names = tuple(class_names)
    print('class_names:', class_names)
    out_class_names_file = osp.join(args.output_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)

    colormap = labelme.utils.label_colormap(255)

    # glob：可以将某目录下所有跟通配符模式相同的文件放到一个列表中，不区分大小写
    for label_file in glob.glob(osp.join(args.input_dir, '*.json')):
        print('Generating dataset from:', label_file)
        with open(label_file) as f:
            # basename 返回文件名
            # splitext 作用是分离文件名与扩展名，返回一个元组
            base = osp.splitext(osp.basename(label_file))[0]
            out_img_file = osp.join(
                args.output_dir, 'JPEGImages', base + '.jpg')
            out_lbl_file = osp.join(
                args.output_dir, 'SegmentationClass', base + '.npy')
            out_png_file = osp.join(
                args.output_dir, 'SegmentationClassPNG', base + '.png')
            out_viz_file = osp.join(
                args.output_dir, 'SegmentationClassVisualization', base + '.jpg')

            # json.load：从json文件中读取字典，json.loads：从json字符中读取字典
            data = json.load(f)

            # dirname 返回目录名
            # 原始图片路径
            img_file = osp.join(osp.dirname(label_file), data['imagePath'])
            # asarray：image 转换成 array
            # np.array(默认情况下)将会copy该对象, np.asarray除非必要否则不会copy该对象
            img = np.asarray(PIL.Image.open(img_file))
            # fromarray：array 转换成 image
            # save：保存原始图
            PIL.Image.fromarray(img).save(out_img_file)

            # 批量将labelme的语义分割标注数据转换为图片
            lbl = labelme.utils.shapes_to_label(
                img_shape=img.shape,
                shapes=data['shapes'],
                label_name_to_value=class_name_to_id,
            )
            # 保存语义分割图
            labelme.utils.lblsave(out_png_file, lbl)
            # 以.npy格式将数组保存到二进制文件中
            np.save(out_lbl_file, lbl)

            # 绘制标签
            viz = labelme.utils.draw_label(lbl, img, class_names, colormap=colormap)
            # 保存 语义分割+原图 叠加图
            PIL.Image.fromarray(viz).save(out_viz_file)


if __name__ == '__main__':
    main()
