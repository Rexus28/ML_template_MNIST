#!/bin/sh

python src/get_MNIST_data.py 
python src/split_and_create_folds.py
python src/train.py --model random_forest --metric sweep --save
python src/test.py --model random_forest --metric sweep

