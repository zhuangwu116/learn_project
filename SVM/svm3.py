from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

print(__doc__)

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70,resize=0.4)

n_samples,h,w=lfw_people.images.shape

X = lfw_people.data
n_features=X.shape[1]

y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")

print("n_samples: %d"%n_samples)

print("n_features: %d"%n_features)
print("n_classes: %d"%n_classes)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

n_components = 150

print("Extracting the top %d eigenfaces from %d faces" % (n_components,X_train.shape[0]))

t0= time()
pca = RandomizedPCA(n_components=n_components,whiten=True).fit(X_train)

print("done in %0.3fs" % (time()-t0))

eigenfaces = pca.components_.reshape((n_components,h,w))

print("Projecting the input data on the eigenfaces orthonormal basis")

t0 = time()

X_train_pca=pca.transform(X_train)

X_test_pca = pca.transform(X_test)

print("done in %0.3fs" % (time()-t0))

print("Fitting the classifier to the traning set")

t0 = time()

param_grid = {'C':[1e3,5e3,1e4,1e5],'gamma':[0.0001,0.0005,0.001,0.01,0.1],}

clf = GridSearchCV(SVC(kernel='rbf',class_weight='auto'),param_grid)

clf = clf.fit(X_train_pca,y_train)

print("done in %0.3fs"%(time()-t0))

print("Best estimatar_found by grid search:")

print(clf.best_estimator_)

print("Predicting people's names on the test set")

t0=time()

y_pred=clf.predict(X_test_pca)

print("done in %0.3fs" % (time()-t0))

print(classification_report(y_test,y_pred,target_names=target_names))

print(confusion_matrix(y_test,y_pred,labels=range(n_classes)))

def plot_gallery(images,titles,h,w,n_row=3,n_col=4):
    plt.figure(figsize=(1.8*n_col,2.4*n_row))
    
    plt.imshow(images[i].reshape((h,w)),cmap=plt.cm.gray)
    
    plt.title(title[i],size=12)
    
    plt.xticks(())
    
    plt.yticks(())
    
    
def title(y_pred,y_test,target_names,i):
    pred_name = target_names[y_pred[i].rsplit(' ',1)[-1]
    
    true_name = target_names[y_test[i].rsplit(' ',1)[-1]
    
    return 'predicted:%s\ntrue:       %s' % (pred_name,true_name)
    
prediction_titles = [title(y_pred,y_test,target_names,i) for i in range(y_pred.shape[0])]

plot_gallery(X_test,prediction_titles,h,w)    

eigenface_titles=["eigenface %d" % i for i in range(eigenfaces.shape[0])]

plot_gallery(eigenfaces,eigenface_titles,h,w)

plt.show()