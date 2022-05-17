import torch
import pandas as pd
import os
from torch.utils.data import Dataset
from skimage import io

class init_dataset(Dataset):
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

        return (image, y_label)

def __input__():
    label_df = pd.read_csv("train.csv")
    label_df['label'][label_df['label'] == '10-1'] = 10
    label_df['label'][label_df['label'] == '10-2'] = 0
    label_df['label'] = label_df['label'].apply(lambda x: int(x))

    label_df.to_csv("train.csv", index=False)


def __output__():
    submission = pd.read_csv("submission.csv")

    submission['label'][submission['label'] == 10] = '10-1'
    submission['label'][submission['label'] == 0] = '10-2'
    submission['label'] = submission['label'].apply(lambda x: str(x))
    submission.to_csv('submission.csv',index=False)

#__input__()
#__output__()