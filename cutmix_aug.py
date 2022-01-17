import uuid
import pandas as pd
import numpy as np

# delete for notebook
import cv2
from utils.lightning.train_modules import LightningTrainModule
from utils.common.cutmix import Cutmix
from collections import Counter
import random
import os
from tqdm import tqdm

import warnings
warnings.simplefilter('ignore')

def cutmix_main():
    lt = LightningTrainModule()
    cutmix = Cutmix()

    train_file, test_file, sample_file, _ = lt.load_csv()

    img_dir = lt.paths.get_data_folder('train')
    cutmix_dir = lt.paths.get_data_folder('cutmix')
    os.makedirs(cutmix_dir, exist_ok=True)

    counter = Counter(train_file['Pawpularity'])
    counter_list = list(counter.values())
    counter_max = max(counter_list)

    cutmix_df = train_file.copy()
    cutmix_df = cutmix_df[['Id', 'Pawpularity']]
    for i in tqdm(range(100)):
        score = i+1
        class_df = train_file[train_file['Pawpularity'] == score]
        idx_range = len(class_df)
        class_df.reset_index(drop=True, inplace=True)

        if counter_max <= idx_range:
            continue

        idx_list = [x for x in range(idx_range)]
        cnt = idx_range
        while cnt < counter_max:
            idx1, idx2 = random.sample(idx_list, 2)

            form1 = f'{class_df.loc[idx1, "Id"]}.jpg'
            img_path1 = os.path.join(img_dir, form1)
            image1 = cv2.imread(img_path1)
            label1 = class_df.loc[idx1, 'Pawpularity']

            form2 = f'{class_df.loc[idx2, "Id"]}.jpg'
            img_path2 = os.path.join(img_dir, form2)
            image2 = cv2.imread(img_path2)
            label2 = class_df.loc[idx2, 'Pawpularity']

            cutmix_image, cutmix_label = cutmix(image1, label1, image2, label2)

            prefix = str(score).zfill(3)
            cutmix_file_name = f'cutmix_{prefix}_{cnt}.jpg'
            cutmix_save_path = os.path.join(cutmix_dir, cutmix_file_name)

            cv2.imwrite(cutmix_save_path, cutmix_image)
            add_data = {'Id': [cutmix_file_name], 'Pawpularity': [cutmix_label]}
            add_df = pd.DataFrame(data=add_data)
            cutmix_df = cutmix_df.append(add_df, ignore_index=True)
            cutmix_df.to_csv(os.path.join(cutmix_dir, f'cutmix_train.csv'), index_label=False)

            cnt += 1


if __name__ == '__main__':
    cutmix_main()