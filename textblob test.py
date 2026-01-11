# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:50:26 2023

@author: 16464
"""
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
import numpy as np

clf = RandomForestClassifier(random_state=0)
X = [[ 1,  2,  3],[11, 12, 13]] # 2 samples 3 features
y = [0, 1]  # classes of each sample
clf.fit(X, y)
RandomForestClassifier(random_state=0)

clf.predict(X)

text = "Donald trump sucks! I hate him! He is the worst person ever" 
text = "Bunch of clowns!!! 🤣🤣🤣" #not detected as being negative

# the closer the subjectivity score is to 1, the more opinionated it is
# subjectivity scores of 0 are OBJECTIVE
# 

blob = TextBlob(text)
print(blob.sentiment)


# polarity = blob.sentiment.polarity
# subjectivity = blob.sentiment.subjectivity

