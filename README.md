# Kaggle-PetFinder

URL: https://www.kaggle.com/c/petfinder-pawpularity-score


## Abstract  
Implemented in pytorch-lightning  
Private LB : 866/3538  
Model : SwinTransformer + MLP head for regression


## Train
```
$ python ./src/train.py
```

You can edit train parameters in ./src/config/config.yaml

## Predict
```
$ python ./src/inference.py
```
