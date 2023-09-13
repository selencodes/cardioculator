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

model = joblib.load(r'C:\Users\fatse\final_model.pkl')

st.title("Kardiyovasküler Hastalık Riskinizi Hesaplayın")

# Kullanıcıdan giriş al
yas = st.number_input("Yaşınızı girin:", step = 1)
kolesterol = st.number_input("Kolesterol değerinizi girin:", step=1, value = 0)
sistolik = st.number_input("Büyük (Sistolik) tansiyon değerinizi girin:", step =1, value = 0)
diastol = st.number_input("Küçük (Diyastolik) tansiyon değerinizi girin :", step =1, value = 0)
boy =st.number_input("Boyunuzu cm cinsinden giriniz :", min_value=1, step =1)
kilo = st.number_input("Kilonuzu kg cinsinden giriniz :", min_value=1, step =1)

bmi = kilo *10000 / (boy**2)
if sistolik < 120 and diastol < 80:
        kat =  0
elif sistolik < 130 and diastol < 85:
        kat = 1
elif (sistolik >= 130 and sistolik <= 139) or (diastol >= 85 and diastol <= 89):
        kat = 2
elif (sistolik >= 140 and sistolik <= 159) or (diastol >= 90 and diastol <= 99):
        kat = 3
elif (sistolik >= 160 and sistolik <= 179) or (diastol >= 100 and diastol <= 109):
        kat = 4
elif sistolik >= 180 or diastol >= 110:
        kat = 5
elif sistolik >= 140 and sistolik <= 160 and diastol < 90:
        kat = 6
elif sistolik > 160 and diastol < 90:
        kat = 7
else:
        kat = -1

if kolesterol <= 200:
    kol = 1
elif kolesterol == range(200,240):
    kol = 2
elif kolesterol >=240:
    kol = 3




data = {'blood_pressure_category': kat,
    'ap_hi': sistolik,
    'ap_lo':diastol,
    'age':yas,
    'cholesterol':kol,
    'bmi':bmi}


def show_recommendations():
    st.write("""
    Öneriler:
    * DÜzenli olarak günde en az 30 dakika yürüyüş yapın.
    * Yağlı besinleri, kızartma yiyecekleri, fast food tarzı yeme düzenini  bırakın.  
    * Düzenli egzersiz yapın.
    * Sigara ve alkol gibi alışkanlıklarınızı bırakın.
    * Stresten uzak durmaya çalışın.
    * Su tüketiminize dikkat edin.
    """)

def good_health():
    st.write("""Sağlığınıza dikkat etmeye devam edin. 
    Kalp hastalıklarından kaçınmak için doğru yoldasınız gibi görünüyor. """)




# Tahmin butonu
if st.button("Tahmin Et"):
    # Veriyi model için uygun formata getir (bu örnekte basit bir liste)
    user = pd.DataFrame([data])
    prediction = model.predict(user)
    if prediction[0]==1:
        st.write("Kardiyovasküler hastalık geçirme riskiniz yüksek.")
        show_recommendations()
    else :
        st.write("Kardiyovasküler hastalık geçirme riskiniz düşük.")
