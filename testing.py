import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

#IMPORTING REQUIRED LIBRARIES
#nltk.download()
import numpy as np
import pandas as pd
import re
import certifi
#import nltk
#nltk.download('all')
from nltk.corpus import stopwords
import seaborn as sns
import pickle
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.naive_bayes import  GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

#%matplotlib inline


#DOWNLOADING NLTK PACKAGES

#nltk.download('stopwords')
#nltk.download('punkt')
stop = stopwords.words('/Users/liviagonsalves/Desktop/NLP_proj/server/helper/english-new')

#IMPORTING DATA AND VISUALIZING
main = pd.read_csv("/Users/liviagonsalves/Desktop/NLP_proj/data/marathi_data.csv")
main.head()
sns.countplot(x='rating', data=main)
data = pd.read_csv("/Users/liviagonsalves/Desktop/NLP_proj/data/marathi_data_translated.csv")
data.head()

#PREPROCESSING DATA
data = data[["marathi","english","rating"]]
data.rename(columns = {'rating':'sentiments'}, inplace = True) 
data.head()
data['english']=data['english'].str.lower()
data['english']=data['english'].str.replace('\W+'," ")
data.head()

#REMOVING STOPWORDS
removed_stop_words = []
for i in data['english']:
    removed_stop_words.append(' '.join([word for word in i.split() if word not in stop]))
data['removed_stopwords'] = removed_stop_words

print(data.head())

#TRAINING NLP MODELS
#Creating features for training the NLP model
x = data.iloc[:, 3].values # Sentences translated from marathi
y = data.iloc[:, 2].values # Sentiment Classes (Sad (-1) Neutral (0) Happy (1))
tfidfconverter = TfidfVectorizer(max_features=200, min_df=1, max_df=0.10)  
x = tfidfconverter.fit_transform(data['removed_stopwords']).toarray()

#Train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=200)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

#Logistic Regression Model
def LR(x_train, x_test, y_train, y_test):
    reg = LogisticRegression()   
    reg = reg.fit(x_train, y_train) 
    pickle.dump(reg,open('/Users/liviagonsalves/Desktop/NLP_proj/training/models/lr.pkl','wb'))
    y_pred = reg.predict(x_test) 
    return y_pred
#Decision Tree model
def DT(x_train, x_test, y_train, y_test):
    tree = DecisionTreeClassifier()
    tree = tree.fit(x_train, y_train)
    pickle.dump(tree,open('/Users/liviagonsalves/Desktop/NLP_proj/training/models/dt.pkl','wb'))
    y_pred = tree.predict(x_test)
    return y_pred
#Naive Bayes Model
def GNB(x_train, x_test, y_train, y_test):
    gnb = GaussianNB()
    gnb = gnb.fit(x_train, y_train)
    pickle.dump(gnb,open('/Users/liviagonsalves/Desktop/NLP_proj/training/models/gnb.pkl','wb'))
    y_pred = gnb.predict(x_test)
    return y_pred
#k-nearest neighbours model
def KNN(x_train, x_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
    knn.fit(x_train, y_train)
    pickle.dump(knn,open('/Users/liviagonsalves/Desktop/NLP_proj/training/models/knn.pkl','wb'))
    y_pred = knn.predict(x_test)
    return y_pred
#Random Forest Classifier model
def RFC(x_train, x_test, y_train, y_test):
    rfc_classifier = RandomForestClassifier(n_estimators=200, random_state=0)  
    rfc_classifier = rfc_classifier.fit(x_train, y_train)
    pickle.dump(rfc_classifier,open('/Users/liviagonsalves/Desktop/NLP_proj/training/models/rfc.pkl','wb'))
    y_pred = rfc_classifier.predict(x_test)
    return y_pred
#Support vector machine model
def SVM(x_train, x_test, y_train, y_test):
    svc_classifier = SVC(kernel='linear')
    svc_classifier = svc_classifier.fit(x_train, y_train)
    pickle.dump(svc_classifier,open('/Users/liviagonsalves/Desktop/NLP_proj/training/models/svm.pkl','wb'))
    y_pred = svc_classifier.predict(x_test)
    return y_pred

#Creating Models
ml_classifiers = ['Logistic Regression Model','Decision Tree Model','Gaussian Naive Bayes Model','KNN Model','Random Forest Model','SVM Model']
function_list = [LR,DT,GNB,KNN,RFC,SVM]
tabulating = []
for i in range(len(ml_classifiers)):
    print("======================================================================================")
    print(f"Creating {ml_classifiers[i]}")
    y_pred = function_list[i](x_train, x_test, y_train, y_test)
    print(confusion_matrix(y_test,y_pred))  
    print(classification_report(y_test,y_pred))  
    acc = accuracy_score(y_test, y_pred)*100
    tabulating.append([ml_classifiers[i],acc])
    print(f"{ml_classifiers[i]} Accuracy >>>> {acc}")
    print("======================================================================================")
    print("")

#Tabulating our final results    
from tabulate import tabulate
headers = ["Models", "Accuracy"]
print(tabulate(tabulating,headers=headers))

#Testing on the saved models
from googletrans import Translator
tr = Translator()
inp = input("Enter a statement in marathi >>>> ")
eng = tr.translate(inp).text
eng = eng.lower().replace('\W+'," ")

removed_stopword = []
for word in eng.split(): 
    if word not in stop:
        removed_stopword.append(word)

eng = np.array([" ".join(removed_stopword)])
print(eng)
x = tfidfconverter.transform(eng).toarray()
print(x)
models = ['lr.pkl','dt.pkl','gnb.pkl','knn.pkl','rfc.pkl','svm.pkl']
for i in models:
    model=pickle.load(open('/Users/liviagonsalves/Desktop/NLP_proj/training/models/'+i,'rb'))
    print(i, " " ,model.predict(x))

#Saving the tfidf vector model for frontend    
pickle.dump(tfidfconverter,open('/Users/liviagonsalves/Desktop/NLP_proj/training/models/tfidf.pkl','wb'))
