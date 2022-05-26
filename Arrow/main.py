<<<<<<< HEAD
import torch
import pandas as pd
import os
from torch.utils.data import Dataset
from skimage import io
from skimage.transform import resize
from PIL import Image


class runse(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)

        y_label = torch.tensor(self.annotations.iloc[index, 1])

        if self.transform:
            image = self.transform(image)
        #print(img_path, y_label)
        return image, y_label, img_path


def __excel__():
    file_list = os.listdir('C:/Download/runes')
    # for i in file_list:
    #     img = Image.open('C:/Download/runes/all/' + i).convert('L')
    #     img.save('C:/Download/runes/' + i)

    png_list = [i for i in file_list if i[len(i) - 4:] == '.png']
    label_list = [['do', 'le', 'ri', 'up'].index(i[:2]) for i in png_list]
    csv_dir = os.path.join('C:/Download/runes', 'runes.csv')

    df = pd.DataFrame({'file': png_list, 'label': label_list})
    df.to_csv(csv_dir, index=False)
#__excel__()
=======
import torch
import pandas as pd
import os
from torch.utils.data import Dataset
from skimage import io
import cv2
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D
#from PLI import Image

from skimage.transform import resize
from PIL import Image


class init_dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        # try:
        #     image = Image.open(img_path)
        #     image.verify()
        # except (IOError, SyntaxError) as e:
        #     print('Bad file: ', img_path)
        #     return
        image = io.imread(img_path, plugin='matplotlib')
        #image = cv2.imread(img_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


def __excel__():
    # file_list0 = "D:/Training/WINTEC/Train_new/0.aug_black_hole"
    # file_list1 = "D:/Training/WINTEC/Train_new/1.aug_white_hole"
    # file_list2 = "D:/Training/WINTEC/Train_new/2.black dots"
    # file_list3 = "D:/Training/WINTEC/Train_new/3.black_hole"
    # file_list4 = "D:/Training/WINTEC/Train_new/4.ND_"
    # file_list5 = "D:/Training/WINTEC/Train_new/5.blur"
    # file_list6 = "D:/Training/WINTEC/Train_new/6.bug"
    # file_list7 = "D:/Training/WINTEC/Train_new/7.WK_"
    # file_list8 = "D:/Training/WINTEC/Train_new/8.cri_pd"
    # file_list9 = "D:/Training/WINTEC/Train_new/9.cris"
    # file_list10 = "D:/Training/WINTEC/Train_new/10.mid_big_hole"
    # file_list11 = "D:/Training/WINTEC/Train_new/11.scratch"
    # file_list12 = "D:/Training/WINTEC/Train_new/12.short scratch"
    # file_list13 = "D:/Training/WINTEC/Train_new/13.squeezed"
    # file_list14 = "D:/Training/WINTEC/Train_new/14.white_hole"
    # all_list = [file_list0, file_list1, file_list2, file_list3, file_list4, file_list5, file_list6, file_list7, file_list8, file_list9, file_list10, file_list11, file_list12, file_list13, file_list14]
    # csv_dir = os.path.join("D:/Training/WINTEC","label.csv")
    # file_list0 = "D:/Training/WINTEC/Train_7/1.CRI"
    # file_list1 = "D:/Training/WINTEC/Train_7/2.HOLE"
    # file_list2 = "D:/Training/WINTEC/Train_7/3.ND"
    # file_list3 = "D:/Training/WINTEC/Train_7/4.PND"
    # file_list4 = "D:/Training/WINTEC/Train_7/5.SCR"
    # file_list5 = "D:/Training/WINTEC/Train_7/6.ST"
    # file_list6 = "D:/Training/WINTEC/Train_7/7.WK"

    file_list0 = "D:/Training/WINTEC/insp_test1/1.CRI"
    file_list1 = "D:/Training/WINTEC/insp_test1/2.HOLE"
    file_list2 = "D:/Training/WINTEC/insp_test1/3.ND"
    file_list3 = "D:/Training/WINTEC/insp_test1/4.PND"
    file_list4 = "D:/Training/WINTEC/insp_test1/5.SCR"
    file_list5 = "D:/Training/WINTEC/insp_test1/6.ST"
    file_list6 = "D:/Training/WINTEC/insp_test1/7.WK"
    csv_dir = os.path.join('D:/Training/WINTEC',"label_test_7.csv")
    all_list = [file_list0,file_list1,file_list2,file_list3,file_list4,file_list5,file_list6]

    df = pd.DataFrame(columns=["file", "label"])

    #print(df)
    for i in range(len(all_list)):
        bmp_list = os.listdir(all_list[i])
        bmp_list = [all_list[i] + '/' + j for j in bmp_list]
        #label_list = [i for j in range(len(bmp_list))]
        for j in bmp_list:
            img = cv2.imread(j,cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(j,img)
        #print(label_list)
        #data = pd.DataFrame({'file': bmp_list, 'label' : label_list})
        #print(data)
        #df = pd.concat([df, data])
    #df.to_csv(csv_dir, index=False)




    # for i in file_list:
    #     img = Image.open('C:/Download/runes/all/' + i).convert('L')
    #     img.save('C:/Download/runes/' + i)
    # png_list = [i for i in file_list if i[len(i) - 4:] == '.png']
    # label_list = [['do', 'le', 'ri', 'up'].index(i[:2]) for i in png_list]
    # csv_dir = os.path.join('C:/Download/runes', 'runes.csv')
    #
    # df = pd.DataFrame({'file': png_list, 'label': label_list})
    # df.to_csv(csv_dir, index=False)
__excel__()
>>>>>>> cdf96ea6a41f66fbfdeb1057def4173cc5f085fc
