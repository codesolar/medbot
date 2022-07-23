from flask import Flask, render_template, json, request
import pickle
import pandas as pd
import numpy as np
#from sklearn.preprocessing import LabelEncoder
from model import imp_feature_names, rf_with_imp_features, labelencoder
app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

# def print_symptom(x):
# print("Symptoms"+x)
# return x


def formatChange(x):
    str = ""
    for ele in x:
        if ele != '[' and ele != ']':
            str += ele
    return str


@app.route('/')
def index():
    return render_template('index.html', data=imp_feature_names)


@app.route('/symptoms')
def check():
    symptoms_present = request.args.getlist("data[]")
    #labelencoder = LabelEncoder()
    test_df = pd.DataFrame([symptoms_present], columns=imp_feature_names)
    disease_assumed = rf_with_imp_features.predict(test_df)
    disease_assumed = labelencoder.inverse_transform(disease_assumed)
    # print(disease_assumed)
    #print("The symptoms are")
    # for x in symptoms_present:
    #   print("symptom is"+x)
    return formatChange(np.array_str(disease_assumed))
    # return "0"


if __name__ == "__main__":
    app.run()
