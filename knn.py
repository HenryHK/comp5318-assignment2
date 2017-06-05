import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import time

for i in range(10):
    sample = np.loadtxt('data/train-sample-'+str(i)+'.csv', delimiter=',')
    target = np.loadtxt('data/train-target-'+str(i)+'.csv', delimiter=',')
    predict_sample = np.loadtxt('data/predict-sample-'+str(i)+'.csv', delimiter=',')
    predict_target = np.loadtxt('data/predict-target-'+str(i)+'.csv', delimiter=',')

    begin = time.time()

    knn = KNeighborsClassifier()
    knn.fit(sample, target)

    result = knn.predict(predict_sample)

    end = time.time()


    result_metrics = metrics.classification_report(result, predict_target)

    with open('result/knn.txt', 'a') as output:
        output.write(result_metrics)
        output.write("\n")
        output.write("Use: "+str(end-begin)+"s")
        output.write('------------------------------------\n')
