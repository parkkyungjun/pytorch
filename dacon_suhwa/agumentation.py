import cv2
import pandas as pd
from torchvision import datasets
import torchvision.transforms as transforms
from ddddd import init_dataset
import numpy as np
import os
def hflip():
    data = pd.read_csv('train.csv')

    # for i in range(1, len(data)):
    #    img_path = os.path.join('train/', data.iloc[i, 0])
    #    img = cv2.imread(img_path)
    #    cv2.flip(img, 1, img)
    #    cv2.imwrite(f'flip/{i}.png',img)

    f = data.copy()
    names = [str(i) + '.png' for i in range(858)]
    f['file_name'] = names
    f.to_csv('file.csv', index=False)
# test_p = [0.9793, 0.935482, 0.985786, 0.891852, 0.889089, 0.975497, 0.525482, 0.23712, 0.103281, 0.0910993, 0.51692, 0.083248, 0.0860751, 0.519434]
# test_n = ["[13]_3.bmp", "[15]_2.bmp", "[1]_2.bmp", "[7]_2.bmp", "[11]_1.bmp", "[12]_1.bmp", "[18]_2.bmp", "[2]_3.bmp", "[4]_1.bmp", "[5]_2.bmp", "[6]_1.bmp", "[8]_1.bmp", "[8]_2.bmp", "[9]_2.bmp"]
# train_p = [0.984235, 0.983578, 0.983217, 0.987069, 0.977011, 0.978782, 0.984015, 0.983921, 0.963586, 0.980493, 0.986635, 0.987077, 0.984661, 0.979523, 0.980437, 0.980881, 0.983797, 0.983994, 0.90354, 0.888731, 0.955379, 0.947298, 0.961513, 0.47095, 0.494443, 0.230505, 0.107181, 0.217867, 0.147641, 0.145986, 0.24568, 0.246917, 0.227888, 0.250748, 0.142223, 0.235966, 0.154732, 0.763305, 0.775812, 0.224407, 0.39293, 0.537121]
# train_n = ["[10]_1.bmp", "[10]_2.bmp", "[10]_3.bmp", "[13]_1.bmp", "[13]_2.bmp", "[14]_1.bmp", "[14]_2.bmp", "[14]_3.bmp", "[15]_1.bmp", "[15]_3.bmp", "[16]_1.bmp", "[16]_2.bmp", "[16]_3.bmp", "[17]_1.bmp", "[17]_2.bmp", "[17]_3.bmp", "[1]_1.bmp", "[1]_3.bmp", "[7]_1.bmp", "[11]_2.bmp", "[11]_3.bmp", "[12]_2.bmp", "[12]_3.bmp", "[18]_1.bmp", "[18]_3.bmp", "[19]_1.bmp", "[19]_2.bmp", "[19]_3.bmp", "[2]_1.bmp", "[2]_2.bmp", "[3]_1.bmp", "[3]_2.bmp", "[3]_3.bmp", "[4]_2.bmp", "[4]_3.bmp", "[5]_1.bmp", "[5]_3.bmp", "[6]_2.bmp", "[6]_3.bmp", "[8]_3.bmp", "[9]_1.bmp", "[9]_3.bmp"]
#
# a = pd.DataFrame({'test_p' : test_p, 'test_n' : test_n})
# b = pd.DataFrame({'train_p': train_p, 'train_n' : train_n})
#
# a.to_csv('asd.csv',index=False)
# b.to_csv('bsd.csv',index=False)
# a = pd.read_csv('suhwa_5.csv')
# for i in range(len(a)):
#     if a['label'][i] == '10-1':
#         a['label'][i] = 10
#     elif a['label'][i] == '10-2':
#         a['label'][i] = 0
#     else:
#         a['label'][i] = int(a['label'][i])
#
# a.to_csv('r_test.csv', index=False)
os.environ['KMP_DUPLICATE_LIB_OK']= 'True'


data_transformer = transforms.Compose([transforms.ToTensor(), transforms.Resize(250), transforms.CenterCrop((224, 224))])
train_ds = init_dataset(csv_file='D:/asddas/user_data/train.csv', root_dir='D:/dacon_suhwa/train_299',
                       transform=data_transformer)

# train_ds = init_dataset(csv_file='r_test.csv', root_dir='D:/dacon_suhwa/test_299',
#                            transform=data_transformer)

meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x,_ in train_ds]
stdRGB = [np.std(x.numpy(), axis=(1,2)) for x,_ in train_ds]

meanR = np.mean([m[0] for m in meanRGB])
meanG = np.mean([m[1] for m in meanRGB])
meanB = np.mean([m[2] for m in meanRGB])

stdR = np.mean([s[0] for s in stdRGB])
stdG = np.mean([s[1] for s in stdRGB])
stdB = np.mean([s[2] for s in stdRGB])

print(meanR, meanG, meanB)
print(stdR, stdG, stdB)


