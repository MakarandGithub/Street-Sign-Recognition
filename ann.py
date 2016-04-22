# Load the Pima Indians diabetes dataset from CSV URL
import numpy as np
import urllib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,classification_report
#from sknn.mlp import Classifier, Layer
result_file = open("result_file500.txt", "w")

# URL for the Pima Indians Diabetes dataset (UCI Machine Learning Repository)
#url = "file:///home/nithin/deep-learning/caffe/data/font_style/font_features.html"
#url_test = "file:///home/nithin/deep-learning/caffe/data/font_style/font_features_test.html"
# download the file
#raw_data = urllib.urlopen(url)
#raw_data_test = urllib.urlopen(url_test)
# load the CSV file as a numpy matrix
#dataset = np.loadtxt(raw_data, delimiter=",")
#dataset_test = np.loadtxt(raw_data_test, delimiter=",")
url = "train.html"
url_test = "test.html"
# download the file
raw_data = urllib.urlopen(url)
raw_data_test = urllib.urlopen(url_test)
dataset = np.loadtxt(raw_data, delimiter=",")
dataset_test = np.loadtxt(raw_data_test, delimiter=",")

print(dataset.shape)
print(dataset_test.shape)
print(dataset_test[50])

X = dataset[:,0:4095]
y = dataset[:,4096]
X_test = dataset_test[:,0:4095]
y_test = dataset_test[:,4096]


# separate the data from the target attributes

#print(y)
#print(X)

clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(500), random_state=1)
clf.fit(X, y) 
#prediction

print(clf.predict(X_test[50]))
print(clf.predict(X_test[150]))
result_array=clf.predict(X_test)
countCorrect=0
print(len(result_array))
for i in range(len(result_array)):
	if(result_array[i]==y_test[i]):
		countCorrect=countCorrect+1
	
print(result_array)
#print_array=[0]*len(result_array)

# for i in range(result_array.shape[0]):
# 	if(result_array[i]==1):
# 		count=count+1	
#print(result_array)

print(countCorrect)
print("The efficiency of the classifier without noise is (percent) :")
print((float(countCorrect)/len(y_test))*100)

target_names=['class 0','class 1','class 2','class 3','class 4','class 5','class 6', 'class 7','class 8','class 9','class 10','class 11','class 12','class 13',
'class 14' , 'class 15', 'class 16' , 'class 17', 'class 18' , 'class 19', 'class 20' , 'class 21', 'class 22', 'class 23','class 24','class 25','class 26',
'class 27','class 28', 'class 29','class 30','class 31','class 32','class 33','class 34','class 35','class 36','class 37','class 38','class 39',
'class 40','class 41','class 42','class 43','class 44','class 45','class 46','class 47','class 48','class 49','class 50','class 51','class 52',
'class 53','class 54','class 55','class 56','class 57','class 58','class 59','class 60','class 61']

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


result_file.write(classification_report(y_test,result_array,target_names=target_names))
# Compute confusion matrix
cm = confusion_matrix(y_test, result_array)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()
