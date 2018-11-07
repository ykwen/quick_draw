import numpy as np
import matplotlib.pyplot as plt
import json
import os


def check_and_load(file_path, categories):
    '''
    Check whether the category is correct and load json files
    :param file_path: the path to the folder of simplified data
    :param categories: the files chosen to transform
    :return: contents of chosen category
    '''
    files = os.listdir(file_path)
    file_set = set(files)
    contents = []
    for c in categories:
        file = c + '.ndjson'
        if file not in file_set:
            raise ValueError("Category {} not found".format(c))
        contents.append(json.loads(file))
    return contents


def decode_drawing(drawing):
    '''
    Transofrom one drawing to feature points
    :param drawing: several continues [num_strokes, [x_N], [y_N]]
    :return: [number_points, 5]
    '''
    features = []
    num_stroke = len(drawing)
    for ind, stroke in enumerate(drawing):
        stroke_length = len(stroke)
        prev = None
        for i, s in enumerate(stroke):
            # set first point
            if not prev:
                features.append(np.array((0, 0, 1, 0, 0), np.int16))
                prev = [s[0][i], s[1][i]]
            else:
                # set regular point
                if i < stroke_length - 1:
                    features.append(np.array((s[0][i] - prev[0], s[1][i] - prev[1], 1, 0, 0), np.int16))
                # set end of stroke point
                else:
                    # set end of drawing point
                    if ind == num_stroke - 1:
                        features.append(np.array((s[0][i] - prev[0], s[1][i] - prev[1], 0, 0, 1), np.int16))
                        return features
                    else:
                        features.append(np.array((s[0][i] - prev[0], s[1][i] - prev[1], 0, 1, 0), np.int16))
                prev = [s[0][i], s[1][i]]
    return features


def transform_to_sketch(file_path, categories):
    '''
    Transform the simplified drawing data in json files to (x, y, p1, p2, p3) used in sketch RNN
    :param file_path: the path to the folder of simplified data
    :param categories: the files chosen to transform
    :return: [num_category, num_content_in_each, L+1, 5], sequence_length,
            where L is the sequence length for each sample with label
    '''
    contents = check_and_load(file_path, categories)
    transformed = []
    for one_cate in contents:
        one_cat_content = []
        for c in one_cate:
            label = c["word"]
            drawing = c["drawing"]
            features = decode_drawing(drawing)
            one_cat_content.append([np.array(features), label])
        transformed.append(one_cat_content)
    return transformed


def visualize_one_picture(drawing):
    '''
    plot drawing from original data
    :param drawing: drawing
    :return: None
    '''
    for x, y in drawing:
        plt.plot(x, y)
    plt.show()


def visualize_one_transformed(features):
    '''
    plot drawing from transformed data
    :param features: transformed features
    :return: None
    '''
    lines = []
    line = []
    for f in features:
        if f[2] == 1:
            line.append(np.array([f[0], f[1]], np.int16))
        elif f[3] == 1:
            line.append(np.array([f[0], f[1]], np.int16))
            lines.append(np.array(line))
            line = []
        else:
            lines.append(np.array(line))
            break
    for l in lines:
        plt.plot(l[:, 0], l[:, 1])
    plt.show()
