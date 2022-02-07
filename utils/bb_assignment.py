from utils.general_functions import list_files


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