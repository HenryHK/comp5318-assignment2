import collections
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import time
from sklearn.cluster import KMeans
from operator import itemgetter, attrgetter

def e_distance(x, y):
    """
    @param x: point.
    @param y: point.
    
    @return euclidean distance between this point.
    """
    x = np.array(x)
    y = np.array(y)
    return np.linalg.norm(x-y)

for count in range(10):

    sample = np.loadtxt('data/train-sample-'+str(count)+'.csv', delimiter=',')
    target = np.loadtxt('data/train-target-'+str(count)+'.csv', delimiter=',')
    predict_sample = np.loadtxt('data/predict-sample-'+str(count)+'.csv', delimiter=',')
    predict_target = np.loadtxt('data/predict-target-'+str(count)+'.csv', delimiter=',')

    original_length = len(sample)
    print("sample size: "+str(original_length))

    begin = time.time()

    k = 7
    kmeans = KMeans(n_clusters=k).fit(sample)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    combined_data = zip(sample, target)

    sample_with_distance=[]
    target_with_distance=[]

    d = collections.defaultdict(list)
    outlier = 0.95

    i=0
    for each in combined_data:
        label = labels[i]
        center = centroids[label]
        distance = e_distance(each[0], center)
        d[label].append((each, distance))
        i += 1

    sample = []
    target = []

    for key, values in d.items():
        length = len(values)
        values.sort(key= itemgetter(1), reverse = False)
        values = values[0:int(length*outlier)]
        for each in values:
            sample.append(each[0][0])
            target.append(each[0][1])

    preprocessing_end = time.time()

    knn = KNeighborsClassifier()
    knn.fit(sample, target)

    result = knn.predict(predict_sample)

    end = time.time()


    result_metrics = metrics.classification_report(result, predict_target)


    with open('result/knn_with_orc.txt', 'a') as output:
        output.write(result_metrics)
        output.write('\n')
        output.write('preprocessing uses: '+str(preprocessing_end-begin)+"s\n")
        output.write("knn use: "+str(end-preprocessing_end)+"s\n")
        output.write("In total: "+str(end-begin)+"s\n")
        output.write("-------------------------------")