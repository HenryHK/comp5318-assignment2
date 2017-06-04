import collections
from operator import attrgetter, itemgetter

import numpy as np
from sklearn import ensemble, metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


def e_distance(x, y):
    """
    @param x: point.
    @param y: point.
    
    @return euclidean distance between this point.
    """
    x = np.array(x)
    y = np.array(y)
    return np.linalg.norm(x-y)


sample = np.loadtxt('data/train-sample-2.csv', delimiter=',')
target = np.loadtxt('data/train-target-2.csv', delimiter=',')
predict_sample = np.loadtxt('data/predict-sample-2.csv', delimiter=',')
predict_target = np.loadtxt('data/predict-target-2.csv', delimiter=',')

original_length = len(sample)
print("sample size: "+str(original_length))

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

# with open('data/train-sample-1-orc.csv', 'w') as data:
#     for line in sample:
#         data.write(",".join(line))
#         data.write("\n")
# with open('data/train-target-1-orc.csv', 'w') as data:
#     for line in target:
#         data.write(line)
#         data.write("\n")

new_length = len(sample)

print("Remove "+str(original_length-new_length)+" outliers")

rf = ensemble.RandomForestClassifier()
rf.fit(sample, target)

result = rf.predict(predict_sample)
result_metrics = metrics.classification_report(result, predict_target)

with open('result/rf_with_kmeans.txt', 'w') as output:
    output.write(result_metrics)
