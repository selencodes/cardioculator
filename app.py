import pandas as pd
import numpy as np
import joblib
# Visualization Libraries 📊

import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

# Machine Learning Models 🤖

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score, confusion_matrix, classification_report, RocCurveDisplay

import joblib
import lightgbm

import streamlit as st
from PIL import Image


model = joblib.load(r'final_model.pkl')


st.title("Calculate your cardiovascular disease risk with Cardioculator!")


image = Image.open("cardd.jpeg")
col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
with col2:
        st.image(image, width = 200)


yas = st.number_input("Enter your age :", step = 1)
kolesterol = st.number_input("Enter your total cholesterol value from your blood test :", step=1, value = 0)
sistolik = st.number_input("Enter your systolic (higher) blood pressure value in mm Hg. (For example, if your blood pressure is 12/8, it is considered as 120/80 mm Hg, where your systolic pressure is 120) :", step =1, value = 0)
diastolik = st.number_input("Enter your diastolic (lower) blood pressure value in mm Hg. (For example, if your blood pressure is 12/8, it is considered as 120/80 mm Hg, where your diastolic pressure is 80) :", step =1, value = 0)
boy =st.number_input("Enter your height in centimeters :", min_value=1, step =1)
kilo = st.number_input("Enter your weight in kilograms :", min_value=1, step =1)

bmi = kilo *10000 / (boy**2)
if sistolik < 120 and diastolik < 80:
        kat =  0
elif sistolik < 130 and diastolik < 85:
        kat = 1
elif (sistolik >= 130 and sistolik <= 139) or (diastolik >= 85 and diastol <= 89):
        kat = 2
elif (sistolik >= 140 and sistolik <= 159) or (diastolik >= 90 and diastolik <= 99):
        kat = 3
elif (sistolik >= 160 and sistolik <= 179) or (diastolik >= 100 and diastolik <= 109):
        kat = 4
elif sistolik >= 180 or diastolik >= 110:
        kat = 5
elif sistolik >= 140 and sistolik <= 160 and diastolik < 90:
        kat = 6
elif sistolik > 160 and diastolik < 90:
        kat = 7
else:
        kat = -1

if kolesterol <= 200:
    kol = 1
elif kolesterol in range(200,240):
    kol = 2
elif kolesterol >=240:
    kol = 3


data = {'blood_pressure_category': kat,
    'ap_hi': sistolik,
    'ap_lo':diastolik,
    'age':yas,
    'cholesterol':kol,
    'bmi':bmi}


def show_recommendations():
    st.write("""
   Recommendations:
   
* Walk at least 30 minutes daily.
* Avoid fatty foods, fried items, and fast-food habits.
* Engage in regular exercise.
* Quit habits like smoking and alcohol consumption.
* Try to stay away from stress.
* Be mindful of your water intake.
    """)

def good_health():
    st.write(""" Keep taking care of your health. It seems you're on the right track to avoid heart diseases. """)

if st.button("Calculate"):
    user = pd.DataFrame([data])
    prediction = model.predict(user)
    if prediction[0]==1:
        st.subheader("Your cardiovascular disease risk seems higher than usual.")
        show_recommendations()
    else :
        st.subheader("Your current risk for cardiovascular disease appears to be low.")
        good_health()
