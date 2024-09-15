import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import graphviz

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from sklearn.metrics import roc_curve, roc_auc_score

data = pd.read_csv('data/thyroid_data.csv')

# normalize the data/label encoding

mappingYN = {"No": 0, "Yes": 1}
bool_cols = ["Smoking", "Hx Smoking", "Hx Radiotherapy", "Recurred"]

data[bool_cols] = data[bool_cols].map(mappingYN.get)

label_encoder = LabelEncoder()

encode_cols = ["Gender", "Focality", "M"] #cols with only 2 unique values

for col in encode_cols:
    data[col] = label_encoder.fit_transform(data[col])

mappingPE = {"Normal": 0, "Single nodular goiter-left": 2, "Single nodular goiter-right": 1, "Multinodular goiter": 2, "Diffuse goiter": 1}

data["Physical Examination"] = data["Physical Examination"].map(mappingPE)

mappingAD = {"No": 0, "Right": 1, "Left": 1, "Posterior": 2,  "Bilateral": 3, "Extensive": 4}

data["Adenopathy"] = data["Adenopathy"].map(mappingAD)

mappingPA = {"Micropapillary": 0, "Papillary": 1, "Follicular": 2, "Hurthle cell": 3}

data["Pathology"] = data["Pathology"].map(mappingPA)

#one hot encoding

ohe = OneHotEncoder()

feature_array = ohe.fit_transform(data[["Thyroid Function"]]).toarray()

feature_labels = ohe.categories_

feature_labels = np.array(feature_labels).ravel()

features = pd.DataFrame(feature_array, columns=feature_labels)

data = pd.concat([data, features], axis=1)
data = data.drop("Thyroid Function", axis=1)

print(data.head())