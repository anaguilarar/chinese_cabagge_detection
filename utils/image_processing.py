import os
import glob
import random
import itertools

from bs4 import BeautifulSoup

# import tensorflow as tf
import io
import numpy as np
from numpy.lib.function_base import kaiser
from six import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from utils.tf_model_detection import single_image_detection
from utils.plt_functions import plot_single_image, plot_single_image_odlabel
from utils.image_functions import split_image, change_images_contrast, from_array_2_jpg

from tqdm import tqdm


# TODO: fif size is not working
# TODO: create a different class for reading pascal data as images


def get_filename_frompath(imgpath):
    imgfilename = imgpath.split('\\')[-1]

    if imgfilename.endswith('jpg'):
        imgfilename = imgfilename[:-4]

    return imgfilename


def percentage_to_bb(bb, size):
    ymin = int(bb[0] * size[1])  # xmin
    xmin = int(bb[1] * size[0])  # ymin
    ymax = int(bb[2] * size[1])  # xmax
    xmax = int(bb[3] * size[0])  # ymax

    return np.array([[xmin, ymin, xmax, ymax]])


def bb_topercentage(bb, size):
    xmin = bb[0] / size[1]  # xmin
    ymin = bb[1] / size[0]  # ymin
    xmax = bb[2] / size[1]  # xmax
    ymax = bb[3] / size[0]  # ymax

    return np.array([[ymin, xmin, ymax, xmax]])


def list_files(input_path, pattern="xml"):
    """

    :param input_path:
    :param pattern:
    :return:
    """
    # taken from https://www.askpython.com/python/examples/list-files-in-a-directory-using-python

    files = glob.glob(input_path + '**/*' + pattern, recursive=True)
    return [f for f in files if os.path.isfile(f)]


def get_image_path_from_xml(data):
    """

    :param xml_file:
    :return:
    """

    bs_data = BeautifulSoup(data, 'xml')
    path_b4 = bs_data.find_all('path')
    image_b4 = bs_data.find_all('filename')

    return [path_b4[0].text,
            image_b4[0].text]


def get_bbox(b4attribute):
    """

    :param b4attribute:
    :return: list
    """
    return [int(b4attribute.find_all('xmin')[0].text),
            int(b4attribute.find_all('ymin')[0].text),
            int(b4attribute.find_all('xmax')[0].text),
            int(b4attribute.find_all('ymax')[0].text)]


def get_bounding_box_from_xml(data):
    """
    This function only works with pascalIVOC annotation format
    more information in https://roboflow.com/formats/pascal-voc-xml
    :param xml_file:
    :return: list [xmin, ymin, xmax, ymax]
    """
    bs_data = BeautifulSoup(data, 'xml')
    boundingbox_list = bs_data.find_all('bndbox')
    if len(boundingbox_list) > 1:
        bb = []
        for i in range(len(boundingbox_list)):
            bb.append(get_bbox(boundingbox_list[i]))
    else:
        bb = [get_bbox(boundingbox_list[0])]

    return bb


def get_image_orgsize_from_xml(data):
    """

    :param xml_file:
    :return:
    """

    bs_data = BeautifulSoup(data, 'xml')
    size_list = bs_data.find_all('size')

    return [int(size_list[0].find_all('height')[0].text),
            int(size_list[0].find_all('width')[0].text)]


def load_image_into_numpy_array(path, scale_factor=None,
                                size=None):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: a file path.

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """

    # img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(io.open(path, "rb", buffering=0).read()))
    (im_width, im_height) = image.size
    if scale_factor is not None:
        im_width = int(im_width * scale_factor / 100)
        im_height = int(im_height * scale_factor / 100)
        image = image.resize((im_width, im_height))
    if size is not None:
        im_width = size[1]
        im_height = size[0]
        image = image.resize((im_width, im_height))
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def export_masked_image(image, ground_box, filename):
    bb_int = percentage_to_bb(ground_box['detection_boxes'][0],
                              (image.shape[1], image.shape[0]))

    image = Image.fromarray(image)
    image = image.crop(box=bb_int[0])

    image.save(filename)
    print("Image saved: {}".format(filename))


class ImageData:

    @property
    def od_labels(self):
        return LabelData(self).labels

    def to_jpg(self, output_path, data_type=None, size=None, verbose=False) -> None:

        if data_type is not None:
            data_type = [data_type] if type(data_type) != list else data_type
        else:
            data_type = self._augmented_data.keys()

        for datatype in data_type:
            for idimage in range(len(self._augmented_data[datatype]['imgs'])):
                fn = self._augmented_data[datatype]['names'][idimage] + '.jpg'
                fn_path = os.path.join(output_path, fn)
                from_array_2_jpg(self._augmented_data[datatype]['imgs'][idimage],
                                 fn_path,
                                 size=size,
                                 verbose=verbose)

    def split_data_into_tiles(self, nrows=None, ncols=None, data_type=None):

        if data_type is not None:
            data_type = [data_type] if type(data_type) != list else data_type
        else:
            data_type = self._augmented_data.keys()

        data_type = [i for i in data_type if i != 'tiles']
        listimgs = []
        fnlist = []
        for datatype in list(data_type):

            for idimage in range(len(self._augmented_data[datatype]['imgs'])):
                tilesdata = split_image(self._augmented_data[datatype]['imgs'][idimage],
                                        nrows=nrows, ncols=ncols)

                fnlist.append(["{}_tile_{}".format(
                    self._augmented_data[datatype]['names'][idimage],
                    i) for i in range(len(tilesdata))])

                listimgs.append(tilesdata)

        self.imgs_tiled = listimgs

        newdata = {'tiles': {'imgs': list(itertools.chain.from_iterable(listimgs)),
                             'names': list(itertools.chain.from_iterable(fnlist))}}

        self._augmented_data.update(newdata)
        print('{} images were added'.format(len(list(itertools.chain.from_iterable(listimgs)))))

    def constrast_augmentation(self, data_type=None, **kwargs):

        if data_type is not None:
            data_type = [data_type] if type(data_type) != list else data_type
        else:
            data_type = self._augmented_data.keys()

        data_type = [i for i in data_type if i != 'contrast']

        listimgs = []
        fnlist = []
        for datatype in data_type:
            for idimage in range(len(self._augmented_data[datatype]['imgs'])):
                contrastdata, combs = change_images_contrast(
                    self._augmented_data[datatype]['imgs'][idimage],
                    **kwargs)

                fnlist.append([
                    "{}_contrast_{}".format(
                        self._augmented_data[datatype]['names'][idimage],
                        comb) for comb in combs])

                listimgs.append(contrastdata)

        self.imgs_contrasted = listimgs

        newdata = {'contrast': {'imgs': list(itertools.chain.from_iterable(listimgs)),
                                'names': list(itertools.chain.from_iterable(fnlist))}}

        self._augmented_data.update(newdata)
        print('{} images were added'.format(len(list(itertools.chain.from_iterable(listimgs)))))

    def plot_image(self, id_image=0, figsize=(12, 10), add_label=False):
        if add_label:
            vector = []
            imgsize = (self.images_data[id_image].shape[0], self.images_data[id_image].shape[1])
            for j in range(len(self.od_labels[id_image])):
                vector.append(from_yolo_toxy(
                    [float(i) for i in self.od_labels[id_image][j]],
                    imgsize))

            plot_single_image_odlabel(self.images_data[id_image], vector)

        else:

            plot_single_image(self.images_data, id_image, figsize)

    def _read_single_image(self, id_image=0, scale_factor=None, size=None, pattern="\\", pos_id=-2):

        kwargs = {'scale_factor': scale_factor,
                  'size': size}

        if type(id_image) == str:
            id_image = [i for i in range(len(self.jpg_path_files))
                        if id_image in self.jpg_path_files[i]]

        id_image = self.jpg_path_files[id_image].split(pattern)[pos_id:]
        id_image_folder = '_'.join(id_image)
        selectid = 0
        if len(id_image) > 0:
            selectid = -1

        path_img_id = [i for i in self.jpg_path_files if id_image[selectid] in i]

        single_image = load_image_into_numpy_array(path_img_id[0], **kwargs)

        return single_image, id_image_folder

    def _organice_data(self, idx, kwargs):
        img_list = []
        bb_list = []

        img_info = self.single_image(idx, **kwargs)
        for m in range(len(img_info[1])):
            img_list.append(img_info[0][0])
            bb_list.append(img_info[1][m])
        return [img_list, bb_list]

    def multiple_images(self,
                        n_samples=None,
                        scale_factor=None,
                        size=None,
                        shuffle=True,
                        seed=123,
                        start=0,
                        read_images=True):

        kwargs = {'scale_factor': scale_factor,
                  'size': size}
        if n_samples is None:
            n_samples = len(self._path_files)

        list_idx = list(range(len(self._path_files)))
        if shuffle:
            random.seed(seed)
            random.shuffle(list_idx)

        img_list = []
        bb_list = []
        idx_list = []

        for i in tqdm(range(start, n_samples)):

            img_info = self.single_image(list_idx[i], read_images=read_images, **kwargs)
            for m in range(len(img_info[1])):
                img_list.append(img_info[0][0])
                bb_list.append(img_info[1][m])
            idx_list.append(list_idx[i])

        return img_list, bb_list, idx_list

    def single_image(self, idx, read_images=True, **kwargs):

        with open(self._path_files[idx], 'r', encoding="utf8") as f:
            data = f.read()

        # get image name
        bsdata = get_image_path_from_xml(data)
        # get boundary box name
        bbdata = get_bounding_box_from_xml(data)
        # get original_size
        sizedata = get_image_orgsize_from_xml(data)
        gt_boxes = []
        train_images_np = []

        for m in range(len(bbdata)):
            gt_boxes.append(bb_topercentage(bbdata[m], sizedata))
            if read_images:
                train_images_np.append(load_image_into_numpy_array(bsdata[0], **kwargs))
            else:
                train_images_np.append(None)

        return [train_images_np, gt_boxes]

    def read_image_data(self, id_images=None, scale_factor=None, size=None, pattern="\\", pos_id=-2):

        kwargs = {'scale_factor': scale_factor, 'size': size, 'pattern': pattern,
                  'pos_id': pos_id}

        images_list = []
        file_names = []
        for i in tqdm(range(len(id_images))):
            imgdata = self._read_single_image(i, **kwargs)
            images_list.append(imgdata[0])
            file_names.append(imgdata[1])

        return images_list, file_names

    def object_detection(self, model,
                         cat_index,

                         min_score=0.55,
                         fig_size=(15, 20),
                         output_path=None,
                         plot=False
                         ):

        detections = []

        for i in tqdm(range(len(self.images_data))):
            if output_path is not None:
                fn = os.path.join(output_path, "n_{}.jpg".format(self.id_image[i]))

            else:
                fn = None

            detections.append(
                single_image_detection(self.images_data[i],
                                       model,
                                       cat_index,
                                       filename=fn,
                                       min_score=min_score,
                                       fig_size=fig_size,
                                       plot=plot))
        return detections

    def __init__(self,
                 source,
                 id_image=None,
                 image_size=None,
                 scale_percentage=None,
                 pattern='jpg',
                 label_type=None,
                 sep_pattern="\\"):
        """

        :param source:
        :param id_image:
        :param image_size:
        :param scale_percentage:
        :param pattern:
        :param sep_pattern:
        """
        self.imgs_contrasted = None
        self._path_files = None
        self._input_path = source
        if pattern is not None:

            self._path_files = list_files(source, pattern=pattern)
            if label_type == "xml":
                self.jpg_path_files = [i[:-4] + ".jpg" for i in self._path_files]

            if pattern == 'jpg':
                self.jpg_path_files = self._path_files

        # TODO: separete images_data and id_image
        if id_image is not None:
            if id_image == "all":
                id_image = list(range(len(self.jpg_path_files)))
            if type(id_image) != list:
                id_image = [id_image]
            self.jpg_path_files = [self.jpg_path_files[i] for i in id_image]

        else:
            id_image = list(range(len(self.jpg_path_files)))

        self.images_data, self.id_image = self.read_image_data(id_images=id_image,
                                                               scale_factor=scale_percentage,
                                                               size=image_size,
                                                               pattern=sep_pattern, pos_id=-2)

        fn_originals = [get_filename_frompath(self.jpg_path_files[i])
                        for i in range(len(self.jpg_path_files))]

        self._augmented_data = {'original': {'imgs': self.images_data,
                                             'names': fn_originals}}


def from_yolo_toxy(yolo_style, size):
    dh, dw = size
    _, x, y, w, h = yolo_style

    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)

    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1

    return (l, r, t, b)


class LabelData:

    def __init__(self,
                 img_class,
                 label_type="yolo",
                 pattern=None):

        self.labeled_data = None

        if label_type == "yolo":

            source = img_class._input_path

            pattern = 'txt'
            self.labels_path_files = list_files(source, pattern=pattern)

            imgsfilepaths = img_class.jpg_path_files.copy()

            fn = [labelfn.split('\\')[-1][:-4] for labelfn in self.labels_path_files]
            fnorig = [labelfn for labelfn in self.labels_path_files]

            organized_labels = []
            labels_data = []
            idlist = []
            for i, imgfn in enumerate(imgsfilepaths):
                if '\\' in imgfn:
                    imgfn = imgfn.split('\\')[-1]
                if imgfn.endswith('jpg'):
                    imgfn = imgfn[:-4]

                lines = None
                datatxt = None
                if imgfn in fn:
                    datatxt = fnorig[fn.index(imgfn)]
                    with open(datatxt, 'rb') as src:
                        lines = src.readlines()
                    lines = [z.decode().split(' ') for z in lines]
                    idlist.append(i)

                organized_labels.append(datatxt)
                labels_data.append(lines)

            # self.labeled_data = {'images': [img_class._augmented_data['original']['imgs'][i] for i in idlist],
            #                     'yolo_boundary':labels_data,
            #                     'filenames':organized_labels}    

            self.labels = labels_data
            self._path = organized_labels
