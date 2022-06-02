import torch
import pandas as pd
import os
from torch.utils.data import Dataset
from skimage import io
import cv2

def abc():
    an = pd.read_csv('train.csv')
    for index in range(len(an)):
        img_path = os.path.join('train', an.iloc[index, 0])
        image = cv2.imread(img_path)
        image = cv2.resize(image,(299,299),image)
        a = str(an.iloc[index, 0])
        # os.mkdir('test_299')
        cv2.imwrite(f'train_299/{a}', image)


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

        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return image, y_label


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

def __differ__():
    google16 = pd.read_csv("googlenet_16/submission.csv")
    efficient = pd.read_csv("efficientnet_b1_16/submission.csv")
    google32 = pd.read_csv("googlenet_32/submission.csv")
    google16l = pd.read_csv("googlenet_16/submission_last.csv")
    efficientl = pd.read_csv("efficientnet_b1_16/submission_last.csv")
    google32l = pd.read_csv("googlenet_32/submission_last.csv")
    google8 = pd.read_csv("submission.csv")
    google8l = pd.read_csv('suhwa_5.csv')
    asd = pd.read_csv('D:/pythonProject/tyu/save/asd.csv')
    #print(google['label'].dtype, efficient['label'].dtype)
    #result = pd.concat([google16['label'],efficient['label'], google32['label'], google32l['label'], google16l['label'], efficientl['label']], axis=1)
    #result = pd.concat([google8['label'], google8l['label']],axis=1)
    #result.columns = ['google16', 'effi', 'google32', "google32l", "google16l", "effil"]
    #result.columns = ['google8', 'google8l']
    #pd.set_option('display.max_row',500)
    #print(result.loc[result['google8'] != result['google8l']])
    ng = 0
    for i in range(len(google8)):
        if google8['label'][i] != google8l['label'][i]:
            #print(i, google8['label'][i], google8l['label'][i])
            ng += 1
    print(f'Accuracy : {(len(google8)-ng) / len(google8)}%')
    #print(result)
    #result['result'] = 0
    # rr = []
    # for i in range(len(result)):
    #     cnt = {'0': 0, '1': 1, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10-1': 0, '10-2': 0}
    #     for c in result.columns:
    #         #print(result[c][i], c, i)
    #         cnt[result[c][i]] += 1
    #     m = max(cnt.values())
    #     rcnt = {v:k for k, v in cnt.items()}
    #     rr.append(rcnt[m])
    #
    # google16['label'] = rr
    # google16.to_csv("save.csv",index=False)
    #print(google16)



# __differ__()
# __input__()
# __output__()