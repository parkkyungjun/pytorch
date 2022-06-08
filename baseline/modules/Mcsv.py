import os
import pandas as pd

file = os.listdir('../../train/labeled_images')
a = pd.DataFrame({'file_name':file})
a['file_name'] = a['file_name'].apply(lambda x: x[:-4])
a.to_csv('../../train/train.csv', index=False)


