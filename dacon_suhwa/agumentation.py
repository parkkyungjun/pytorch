import cv2
import os
import pandas as pd

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
