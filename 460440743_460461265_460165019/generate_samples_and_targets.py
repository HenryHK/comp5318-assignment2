#!/usr/bin/env python3

from random import shuffle
import csv

values = []

with open('covtype.data', 'r') as data:
    for line in data:
        current = line.split(",")
        values.append(current)

length = len(values)

for i in range(10):
    shuffle(values)
    # filename = 'sample-'+i+'.data'
    train_sample = []
    train_target = []
    predict_sample = []
    predict_target = []
    cutoff = int(length*0.9)
    for line in values[0:cutoff]:
        train_sample.append(line[0:-1])
        train_target.append(line[-1])
    for line in values[cutoff:]:
        predict_sample.append(line[0:-1])
        predict_target.append(line[-1])
    
    # print(predict_sample)

    with open('data/train-sample-'+str(i)+'.csv', 'w') as data:
        for line in train_sample:
            data.write(",".join(line))
            data.write("\n")
    with open('data/train-target-'+str(i)+'.csv', 'w') as data:
        for line in train_target:
            data.write(line)
    with open('data/predict-sample-'+str(i)+'.csv', 'w') as data:
        for line in predict_sample:
            data.write(",".join(line))
            data.write("\n")
    with open('data/predict-target-'+str(i)+'.csv', 'w') as data:
        for line in predict_target:
            data.write(line)

    
