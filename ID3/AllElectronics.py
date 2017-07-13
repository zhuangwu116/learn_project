from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO
import os
CSV_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)),'learn.csv')
print CSV_DIR
allElectronicsData=open(CSV_DIR,'rb')
reader=csv.reader(allElectronicsData)
headers=reader.next()

featureList=[]
labelList=[]

for row in reader:
    labelList.append(row[len(row)-1])
    rowDict={}
    for i in range(1,len(row)-1):
        rowDict[headers[i]]=row[i]
    featureList.append(rowDict)
    
vec=DictVectorizer()
dummyX=vec.fit_transform(featureList).toarray()

print(vec.get_feature_names())

lb=preprocessing.LabelBinarizer()
dummyY=lb.fit_transform(labelList)

clf=tree.DecisionTreeClassifier(criterion='entropy')
clf=clf.fit(dummyX,dummyY)
with open('allElectronicInformationGainOri.dot','w') as f:
    f=tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)
    
oneRowX=dummyX[0,:]
newRowX=oneRowX    
newRowX[0]=1
newRowX[2]=0

predictedY=clf.predict(newRowX)

print predictedY