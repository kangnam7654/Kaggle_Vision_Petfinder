import cv2
import numpy as np
from utils.common.project_paths import GetPaths
import glob

class Cutmix:
    def __init__(self):
        pass

    def __call__(self, image1, label1, image2, label2, Lambda=None):
        self.image1 = image1
        self.label1 = label1
        self.image2 = image2
        self.label2 = label2
        self.pre_process()
        self.Lambda = self.set_lambda(Lambda=Lambda)
        self.x1, self.x2, self.y1, self.y2 = self.set_coordinates()
        self.mix()
        self.adjusted_lambda = self.get_adjusted_lambda()
        self.adjusted_label = self.get_adjusted_label()
        return self.image1, self.adjusted_label

    def pre_process(self):
        if self.image1.shape != self.image2.shape:
            if self.image1.size < self.image2.size:
                reference_shape = self.image1.shape[:2]
            else:
                reference_shape = self.image2.shape[:2]
            self.image1 = cv2.resize(self.image1, reference_shape)
            self.image2 = cv2.resize(self.image2, reference_shape)

    @staticmethod
    def set_lambda(Lambda=None):
        if Lambda is None:
            L = np.random.uniform(0, 1)
        else:
            L = Lambda
        return L

    def set_coordinates(self):
        self.width = self.image1.shape[0]
        self.height = self.image1.shape[1]

        random_x = np.random.uniform(0, self.width)
        random_y = np.random.uniform(0, self.height)
        random_width = self.width * np.sqrt(1 - self.Lambda)
        random_height = self.height * np.sqrt(1 - self.Lambda)

        x1 = np.clip(a=random_x - random_width / 2,
                     a_min=0,
                     a_max=self.width)
        x2 = np.clip(a=random_x + random_width / 2,
                     a_min=0,
                     a_max=self.width)
        y1 = np.clip(a=random_y - random_height / 2,
                     a_min=0,
                     a_max=self.height)
        y2 = np.clip(a=random_y + random_height / 2,
                     a_min=0,
                     a_max=self.height)

        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        return x1, x2, y1, y2

    def mix(self):
        self.image1[self.x1: self.x2, self.y1: self.y2, :] = self.image2[self.x1: self.x2, self.y1: self.y2, :]

    def get_adjusted_lambda(self):
        adjusted_lambda = 1 - (self.x2 - self.x1) * (self.y2 - self.y1) / (self.width * self.height)
        return adjusted_lambda

    def get_adjusted_label(self):
        adjusted_label = self.adjusted_lambda * self.label1 + (1 - self.adjusted_lambda) * self.label2
        return adjusted_label

if __name__ == '__main__':
    paths = GetPaths()
    train_list = glob.glob(paths.get_data_folder('train', '*'))

    cat = cv2.imread('/Users/kimkangnam/Desktop/Project/kaggle/Pawpularity/data/train/0a0da090aa9f0342444a7df4dc250c66.jpg')
    dog = cv2.imread('/Users/kimkangnam/Desktop/Project/kaggle/Pawpularity/data/train/0a0f8edf69eef0639bc2b30ce0cf09d5.jpg')
    cutmix = Cutmix()

    result, label = cutmix(cat, 1, dog, 0)

    print(label)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()