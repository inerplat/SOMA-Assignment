import sys
from sklearn.externals import  joblib
from sklearn.grid_search import GridSearchCV
from sklearn.svm import  LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import  TfidfVectorizer,CountVectorizer
import os
import numpy as np
import string
from keras import backend
from keras.layers import Dense, Input, Lambda, LSTM, TimeDistributed
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
from gensim.models.keyedvectors import KeyedVectors
import json

def isNumber(s):
  try:
    float(s)
    return True
  except ValueError:
    return False

x_text_list = []
y_text_list = []
enc = sys.getdefaultencoding()
with open("refined_category_dataset.dat",encoding=enc) as fin:
    for line in fin.readlines():
#         print (line)
        info = json.loads(line.strip())
        x_text_list.append((info['pid'],info['name']))
        y_text_list.append(info['cate'])
y_name_id_dict = joblib.load("y_name_id_dict.dat")
y_id_list = [y_name_id_dict[x] for x in y_text_list]
from sklearn.model_selection import train_test_split

vectorizer = CountVectorizer()
x_list = vectorizer.fit_transform(map(lambda i : i[1],x_text_list))
y_list = [y_name_id_dict[x] for x in y_text_list]
#print(x_text_list)
tmp_list = list(map(lambda i : i[1],x_text_list))
#print(tmp_list)
from konlpy.corpus import kolaw
from konlpy.tag import *
from konlpy.tag import Kkma
from konlpy.utils import pprint
