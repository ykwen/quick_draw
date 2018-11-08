import numpy as np
import matplotlib.pyplot as plt
import json
import os


def check_and_load(file_path, categories):
    """
    Check whether the category is correct and load json files
    :param file_path: the path to the folder of simplified data
    :param categories: the files chosen to transform
    :return: contents of chosen category
    """
    files = os.listdir(file_path)
    file_set = set(files)
    contents = []
    for c in categories:
        file = file_path + "/" + c + ".ndjson"
        if c + ".ndjson" not in file_set:
            raise ValueError("Category {} not found".format(c))
        with open(file, "r") as f:
            lines = f.readlines()
        contents.append([json.loads(line) for line in lines])
    return contents


def decode_drawing(drawing):
    """
    Transofrom one drawing to feature points
    :param drawing: several continues [num_strokes, [x_N], [y_N]]
    :return: [number_points, 5]
    """
    features = []
    num_stroke = len(drawing)
    prev = [0, 0]
    for ind, s in enumerate(drawing):
        stroke_length = len(s[0])
        for i in range(stroke_length):
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
    """
    Transform the simplified drawing data in json files to (x, y, p1, p2, p3) used in sketch RNN
    :param file_path: the path to the folder of simplified data
    :param categories: the files chosen to transform
    :return: [num_category, 2, num_content_in_each, L, 5], sequence_length,
            where L is the sequence length for each sample with label
    """
    contents = check_and_load(file_path, categories)
    transformed = []
    for one_cate in contents:
        one_cat_content = []
        label = one_cate[0]["word"]
        for c in one_cate:
            drawing = c["drawing"]
            features = decode_drawing(drawing)
            one_cat_content.append(np.array(features))
        transformed.append([one_cat_content, label])
    return transformed


def visualize_one_picture(drawing):
    """
    plot drawing from original data
    :param drawing: drawing
    :return: None
    """
    for x, y in drawing:
        plt.plot(x, y)
    plt.show()


def visualize_one_transformed(features, y=None):
    """
    plot drawing from transformed data
    :param features: transformed features [L, 5]
    :param y: label of data, for fig title
    :return: None
    """
    lines = []
    prev = None
    draw = False
    for f in features:
        if not prev:
            prev = [f[0], f[1]]
            draw = f[2] == 1
            continue
        if draw:
            lines.append([[prev[0], prev[0] + f[0]], [prev[1], prev[1] + f[1]]])
        if f[2] == 1:
            draw = True
        elif f[3] == 1:
            draw = False
        else:
            break
        prev = [f[0] + prev[0], f[1] + prev[1]]
    if y:
        plt.title(y)
    for l in lines:
        plt.plot(l[0], l[1])
    plt.show()


def save_transformed(transformed, save_path):
    for one in transformed:
        label = one[1]
        file = save_path + "/" + label + ".npy"
        np.save(file, one[0])


def load_one_transformed(file):
    return np.load(file)


def get_max_seq_len(data):
    max_l = 0
    for d in data:
        max_l = max(max_l, len(d))
    return max_l


def get_mean_seq_len(data):
    l = []
    for d in data:
        l.append(len(d))
    return np.mean(l)


if __name__ == '__main__':
    file_path = "./data/simplified"
    save_path = "./data/transformed"
    categories = ["cat"]

    transformed = transform_to_sketch(file_path, categories)
    save_transformed(transformed, save_path)

    trans = load_one_transformed(save_path + "/" + "cat.npy")

    contents = check_and_load(file_path, categories)
    for i in range(2):
        visualize_one_transformed(trans[i])
        visualize_one_picture(contents[0][i]["drawing"])

    length = len(contents[0][0]["drawing"][0][0])
    print(length, transformed[0][0][0][length-3: length+1], contents[0][0]["drawing"])
