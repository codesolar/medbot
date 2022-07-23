from copyreg import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import pickle
#from app import symptoms_present
# %matplotlib inline

imported_train = pd.read_csv('project_train.csv')
# imported_train.head()

# imported_train.info()

# denotes total number of distinct diseases this chatbot can check
# len(set(imported_train['prognosis']))

# set(imported_train['prognosis'])  # all diseases

imported_train = imported_train.drop_duplicates()
# imported_train.info()

imported_train.reset_index()
#sns.heatmap(imported_train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

X = imported_train.drop('prognosis', axis=1)
y = imported_train['prognosis']
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=0)

#X_train.shape, X_test.shape, y_train.shape, y_test.shape

# to see balanced or imbalanced dataset
#sns.countplot(x="prognosis", data=imported_train)


def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    # print((corr_matrix))
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j]) >= threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    # print(corr_matrix.iloc[3,0])
    return col_corr


corr_features = correlation(X, 0.8)
# len(set(corr_features))

X_train1 = X_train.drop(corr_features, axis=1)
X_test1 = X_test.drop(corr_features, axis=1)

#dt = DecisionTreeClassifier()
#dt.fit(X_train1, y_train)
#print(dt.score(X_test1, y_test))

# Import the model we are using
# Instantiate model with 50 decision trees
rf = RandomForestClassifier(n_estimators=50, random_state=0)
rf.fit(X_train1, y_train)
rf.score(X_test1, y_test)

feature_list = X_train1.columns
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 5))
                       for feature, importance in zip(feature_list, importances)]
feature_importances.sort(key=lambda x: x[1], reverse=True)
# print(feature_importances)
# print(len(feature_list))
temp_feature_list = []
for f, i in feature_importances:
    if i >= 0.01:
        temp_feature_list.append(f)
imp_feature_names = temp_feature_list
#print("number of important features={}".format(len(imp_feature_names)))

rf_with_imp_features = RandomForestClassifier(n_estimators=50, random_state=0)
X_train_imp_features = X_train1.loc[:, imp_feature_names]
X_test_imp_features = X_test1.loc[:, imp_feature_names]
rf_with_imp_features.fit(X_train_imp_features, y_train)
pickle.dump(rf_with_imp_features, open('model.pkl', 'wb'))
rf_with_imp_features.score(X_test_imp_features, y_test)

#symptoms_present = []
# for i in range(38):
#print("do you have "+imp_feature_names[i])
#val = int(input("yes or no :"))
# if val=="yes":
#   val=1
# else:
#   val=0
# symptoms_present.append(val)
#test_df = pd.DataFrame([symptoms_present], columns=imp_feature_names)
#disease_assumed = rf_with_imp_features.predict(test_df)
#disease_assumed = labelencoder.inverse_transform(disease_assumed)
#print("you may have:", disease_assumed)
