import numpy as np
import pandas as pd


def make_bins(csv):
    num_bins = int(np.floor(1 + (3.3) * (np.log10(len(csv)))))
    # num_bins = int(np.ceil(2*((len(csv))**(1./3))))
    csv['bins'] = pd.cut(csv['Pawpularity'], bins=num_bins, labels=False)
    return csv

