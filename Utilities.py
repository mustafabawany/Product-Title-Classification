import numpy as np
import pandas as pd

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#------ Additional Lib
# import neattext as nt
# import neattext.functions as nfx

from sklearn.feature_extraction.text import TfidfVectorizer   # Turning textual data into numeric for computation
from sklearn.preprocessing import OneHotEncoder               # For encoding categorical target attr
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn import svm   # Baseline

# ------- Validation metrics
from sklearn.metrics import accuracy_score    
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import hamming_loss                  
from sklearn.metrics import classification_report


# nltk.download('stopwords')

def featureSelection(df):
    df.drop(['country','sku_id','price','type'],inplace=True,axis=1) #1 means col wise drop 
    df['titleDescp'] = df['title']+" "+df['description']
    df.drop(['title', 'description'],inplace=True,axis=1) 
    Y1 = df['category_lvl1']
    Y2 = df['category_lvl2']
    Y3 = df['category_lvl3']

    return df,Y1,Y2,Y3

def PreProcessing(content):
    ps = PorterStemmer()
    CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

    # Using str(content) because there are some float values in combined
    stemmed_content = re.sub('[^a-zA-Z]',' ',str(content))   # Dropping all encodings, numbers etc
    stemmed_content = re.sub(CLEANR, '',stemmed_content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content

def Cleaning_Data_Utility(training_df):
    X,Y1,Y2,Y3=featureSelection(training_df) 
    X['titleDescp'] = X['titleDescp'].apply(PreProcessing)
    
    return X,Y1,Y2,Y3
