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
