import streamlit as st 
from sklearn.ensemble import  RandomForestClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
st.write('''
         # This app is about the prediction of iris flower species          
''')
def user_input_parameter():
    data=None
    st.sidebar.title('Input iris  parameters')
    sepal_length=st.sidebar.slider('sepal_length',0,15,5)
    sepal_width=st.sidebar.slider('sepal_width',2.00,4.40,3.40)
    petal_length=st.sidebar.slider('petal_length',1.00,6.90,1.30)
    petal_width=st.sidebar.slider('petal_width',0.10,2.50,0.20)
    data={'sepal_length': sepal_length,
          'sepal_width': sepal_width,
          'petal_length':petal_length,
          'petal_width':petal_width
          }
    features=pd.DataFrame(data,index=[0])
    return features
df=user_input_parameter()
st.header('Iris Selected Parameter')
st.write(df)
iris=sns.load_dataset('iris')
st.header('Iris Dataset')
st.write(iris.sample(10))
X=iris[['sepal_length','sepal_width','petal_length','petal_width']]
y=iris['species']
model =RandomForestClassifier()
model.fit(X,y)
prediction=model.predict(df)
prediction_prob=model.predict_proba(df)
st.subheader('Class labels and its coressponding values')
st.markdown(""" 
            index|speceies|
            |----|--------|
            0|setose|
            1|versicolor|
            2|virginica|""")
st.header('Prediction')
st.write(prediction)
st.subheader('Prediction value')
st.write(prediction_prob)
st.title('Some interractive plots')
st.header('scatter plot')
fig=px.scatter(iris,x='sepal_length',y='sepal_width',color='species')
st.plotly_chart(fig,use_container_width=True)

st.header('Bar plot')
st.plotly_chart(px.bar(iris,x='species',y='sepal_width',color='species'),use_container_width=True)
st.header('Histogram')
st.plotly_chart(px.histogram(iris,x='sepal_length',y='sepal_width',color='species'),use_container_width=True)

st.header('Box plot')
fig=px.box(iris,x='species',y='petal_length',color='species')
st.plotly_chart(fig)
plt.tight_layout()

st.header('Area under the curve')
st.plotly_chart(px.area(iris,x='species',y='petal_width',color='species'))


st.header('Heatmap')
iris1=iris[['sepal_length','sepal_width','petal_length','petal_width']]
iris1=pd.DataFrame(iris1)
# Calculate  correlation matrix
correlation_matrix = iris1.corr()
st.plotly_chart(px.imshow(correlation_matrix))

st.header('Pie chart')
st.plotly_chart(px.pie(iris,names='species',values='sepal_length',color='species'))

st.header("3D-Scatter Plot")
fig=px.scatter_3d(iris,x='sepal_length',y='sepal_width',z='petal_length',color='species',animation_frame=iris['sepal_length'])
st.plotly_chart(fig)


