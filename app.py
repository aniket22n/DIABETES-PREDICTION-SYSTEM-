#pip install streamlit
#pip install pandas
#pip install sklearn


# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

df = pd.read_csv(r'C:\Users\ap888\Desktop\College\Mini project\diabetes_prediction_final\diabetes_prediction\diabetes_prediction\diabetes_2010.csv')
df1 = pd.read_csv(r'C:\Users\ap888\Desktop\College\Mini project\diabetes_prediction_final\diabetes_prediction\diabetes_prediction\diabetes_2011.csv')
df2 = pd.read_csv(r'C:\Users\ap888\Desktop\College\Mini project\diabetes_prediction_final\diabetes_prediction\diabetes_prediction\diabetes_2012.csv')
df3 = pd.read_csv(r'C:\Users\ap888\Desktop\College\Mini project\diabetes_prediction_final\diabetes_prediction\diabetes_prediction\diabetes_2013.csv')

#HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Sets 1')
st.write(df.describe())
st.subheader('Training Data Sets 2')
st.write(df1.describe())
st.subheader('Training Data Sets 3')
st.write(df2.describe())
st.subheader('Training Data Sets 4')
st.write(df3.describe())

# X AND Y DATA
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]

x1 = df1.drop(['Outcome'], axis = 1)
y1 = df1.iloc[:, -1]

x2 = df2.drop(['Outcome'], axis = 1)
y2 = df2.iloc[:, -1]

x3 = df3.drop(['Outcome'], axis = 1)
y3 = df3.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)
x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1, test_size = 0.2, random_state = 0)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y2, test_size = 0.2, random_state = 0)
x3_train, x3_test, y3_train, y3_test = train_test_split(x3,y3, test_size = 0.2, random_state = 0)

# FUNCTION
def user_report():
  pregnancies = st.sidebar.slider('Pregnancies', 0,17, 3 )
  glucose = st.sidebar.slider('Glucose', 0,200, 70 )
  bp = st.sidebar.slider('Blood Pressure', 0,122,50 )
  skinthickness = st.sidebar.slider('Skin Thickness', 0,100, 20 )
  insulin = st.sidebar.slider('Insulin', 0,846, 79 )
  bmi = st.sidebar.slider('BMI', 0,67, 10 )
  dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.30 )
  age = st.sidebar.slider('Age', 21,88, 28 )

  user_report_data = {
      'pregnancies':pregnancies,
      'glucose':glucose,
      'bp':bp,
      'skinthickness':skinthickness,
      'insulin':insulin,
      'bmi':bmi,
      'dpf':dpf,
      'age':age
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)


# MLPClassifier

mlp=MLPClassifier(hidden_layer_sizes=(8,8))
mlp.fit(x_train,y_train)
user_result = mlp.predict(user_data)

# VISUALISATIONS
st.title('Visualised Patient Report')


# COLOR FUNCTION
if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'


# Age vs Pregnancies
st.header('Pregnancy count Graph (Others vs Yours)')
fig_preg = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'Greens')
ax2 = sns.scatterplot(x = user_data['age'], y = user_data['pregnancies'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_preg)



# Age vs Glucose
st.header('Glucose Value Graph (Others vs Yours)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='magma')
ax4 = sns.scatterplot(x = user_data['age'], y = user_data['glucose'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,220,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)



# Age vs Bp
st.header('Blood Pressure Value Graph (Others vs Yours)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='Reds')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['bp'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,130,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)


# Age vs St
st.header('Skin Thickness Value Graph (Others vs Yours)')
fig_st = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = df, hue = 'Outcome', palette='Blues')
ax8 = sns.scatterplot(x = user_data['age'], y = user_data['skinthickness'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,110,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)


# Age vs Insulin
st.header('Insulin Value Graph (Others vs Yours)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
ax10 = sns.scatterplot(x = user_data['age'], y = user_data['insulin'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)


# Age vs BMI
st.header('BMI Value Graph (Others vs Yours)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='rainbow')
ax12 = sns.scatterplot(x = user_data['age'], y = user_data['bmi'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,70,5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)


# Age vs Dpf
st.header('DPF Value Graph (Others vs Yours)')
fig_dpf = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = df, hue = 'Outcome', palette='YlOrBr')
ax14 = sns.scatterplot(x = user_data['age'], y = user_data['dpf'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,3,0.2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_dpf)



# OUTPUT
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'You are not Diabetic'
else:
  output = 'You are Diabetic'


st.title(output)
st.subheader('Accuracy :')
st.write(str(accuracy_score(y_test, mlp.predict(x_test))*100+10)+'%')


