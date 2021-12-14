#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import numpy as np 
import seaborn as sns


# In[62]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[63]:


diabetesdata=pd.read_csv('C:/Users/VICKFURY/Documents/Python Scripts/ml/datasets/diabetes.csv')


# In[64]:


diabetesdata.head()


# In[65]:


diabetesdata.tail()


# In[66]:


diabetesdata.shape


# In[67]:


diabetesdata.describe()


# In[68]:


diabetesdata.isnull().sum()


# In[69]:


diabetesdata.corr()


# In[70]:


plt.figure(figsize=(10,10))
sns.heatmap(diabetesdata.corr(),annot=True)


# In[71]:


sns.pairplot(diabetesdata)


# In[72]:


sns.relplot(x='Pregnancies',y='Insulin',hue='Age',data=diabetesdata)


# In[73]:


plt.figure(figsize=(20,20))


sns.boxplot(data=diabetesdata)


# In[74]:


plt.figure(figsize=(10,10))

sns.distplot(diabetesdata['Pregnancies'])


# In[75]:


sns.catplot(x='Pregnancies',kind='box',data=diabetesdata)


# In[76]:


diabetesdata[(diabetesdata['Pregnancies']>=0 )&(diabetesdata['Pregnancies']<12.5)]


# In[77]:


maxthreshold=diabetesdata['Pregnancies'].quantile(0.95)


# In[78]:


diabetesdata[diabetesdata['Pregnancies']> maxthreshold]


# In[79]:


minthreshold=diabetesdata['Pregnancies'].quantile(0.05)


# In[80]:


diabetesdata[diabetesdata['Pregnancies']< minthreshold]


# In[81]:


plt.hist(diabetesdata.Insulin,bins=5,rwidth=0.8)
# plt.xlabel('Pregnancy Numbers')
# plt.ylabel('Count')
plt.show()


# In[82]:


sns.catplot(kind='box',x='Insulin',data=diabetesdata)


# In[83]:


diabetesdata['Insulin'].quantile(0.95)


# In[84]:


(diabetesdata['Insulin']>293.0).sum()


# In[85]:


diabetesdata['Insulin'].mean()


# # REMOVE OUTLIERS IN INSULIN COLUMN

# In[86]:



Q3=diabetesdata['Insulin'].quantile(0.75)
Q1=diabetesdata['Insulin'].quantile(0.25)
Q3,Q1
IQR=Q3-Q1
IQR


# In[87]:


diabetesdata['Insulin'].quantile(0.95)


# In[88]:


# finding our LOWER limit
# Q1-1.5*IQR
lower_limit=Q1-1.5*IQR
# finding our upper limit
# Q3+1.5*IQR
upper_limit=Q3+1.5*IQR
lower_limit,upper_limit


# In[89]:


diabetesdata[(diabetesdata['Insulin']<lower_limit)|(diabetesdata['Insulin']>upper_limit)]


# In[90]:


upper_limit=260.0
diabetesdata2=diabetesdata[diabetesdata['Insulin']<upper_limit]


# In[91]:


diabetesdata2


# In[92]:


sns.boxplot(x='Insulin',data=diabetesdata2)


# In[96]:



Q3=diabetesdata['Insulin'].quantile(0.75)
Q1=diabetesdata['Insulin'].quantile(0.25)

IQR=Q3-Q1
IQR,Q3,Q1


# In[97]:


# finding our LOWER limit
# Q1-1.5*IQR
lower_limit=Q1-1.5*IQR
# finding our upper limit
# Q3+1.5*IQR
upper_limit=Q3+1.5*IQR
lower_limit,upper_limit


# In[98]:


(diabetesdata.Insulin>upper_limit).sum()


# In[ ]:





# In[99]:


diabetesdata.Insulin=np.where(diabetesdata.Insulin>upper_limit,upper_limit,diabetesdata.Insulin)


# In[100]:


(diabetesdata.Insulin>upper_limit).sum()


# In[101]:


sns.boxplot(x='Insulin',data=diabetesdata)


# # REMOVE OUTLIERS IN Pregnancies COLUMN

# In[102]:


sns.catplot(kind='box',x='Pregnancies',data=diabetesdata)


# In[103]:


Q3=diabetesdata['Pregnancies'].quantile(0.75)
Q1=diabetesdata['Pregnancies'].quantile(0.25)
Q3,Q1


# In[104]:


IQR=Q3-Q1
IQR


# In[105]:


# finding our LOWER limit
# Q1-1.5*IQR
lower_limit=Q1-1.5*IQR
# finding our upper limit
# Q3+1.5*IQR
upper_limit=Q3+1.5*IQR
lower_limit,upper_limit


# In[106]:


(diabetesdata.Pregnancies>upper_limit).sum()


# In[107]:


(diabetesdata.Pregnancies<lower_limit).sum()


# In[108]:


# (diabetesdata.Pregnancies>upper_limit).replace(upper_limit,inplace=True)
diabetesdata.Pregnancies=np.where(diabetesdata.Pregnancies>upper_limit,upper_limit,diabetesdata.Pregnancies)


# In[109]:


(diabetesdata.Pregnancies>upper_limit).sum()


# In[110]:


sns.catplot(kind='box',x='Pregnancies',data=diabetesdata)


# In[111]:


diabetesdata.Pregnancies.unique()


# # OUTLIER REMOVAL IN GLUCOSE

# In[112]:


sns.catplot(kind='box',x='Glucose',data=diabetesdata)


# In[113]:


diabetesdata.Glucose.unique()


# In[114]:


diabetesdata.Glucose.describe()


# In[115]:


Q1=diabetesdata.Glucose.quantile(0.25)
Q3=diabetesdata.Glucose.quantile(0.75)
IQR=Q3-Q1
Q1,Q3,IQR


# In[116]:


upper_limit=Q3+1.5*IQR
lower_limit=Q1-1.5*IQR
upper_limit,lower_limit


# In[117]:


diabetesdata.Glucose=np.where(diabetesdata.Glucose>upper_limit,upper_limit,diabetesdata.Glucose)
diabetesdata[diabetesdata.Glucose>upper_limit]


# In[118]:


diabetesdata.Glucose=np.where(diabetesdata.Glucose<lower_limit,lower_limit,diabetesdata.Glucose)

diabetesdata[diabetesdata.Glucose<lower_limit]


# In[119]:


sns.catplot(kind='box',x='Glucose',data=diabetesdata)


# # BLOODPRESSURE OUTLIERS REMOVAL

# In[120]:


sns.catplot(kind='box',x='BloodPressure',data=diabetesdata)


# In[121]:


Q1=diabetesdata.BloodPressure.quantile(0.25)
Q3=diabetesdata.BloodPressure.quantile(0.75)
IQR=Q3-Q1
IQR,Q1,Q3


# In[122]:


upper_limit=Q3+1.5*IQR
lower_limit=Q1-1.5*IQR
upper_limit,lower_limit


# In[123]:


diabetesdata.BloodPressure=np.where(diabetesdata.BloodPressure>upper_limit,upper_limit,diabetesdata.BloodPressure)
diabetesdata.BloodPressure=np.where(diabetesdata.BloodPressure<lower_limit,lower_limit,diabetesdata.BloodPressure)


# In[124]:


sns.catplot(kind='box',x='BloodPressure',data=diabetesdata)


# # AGE OUTLIER REMOVAL

# In[125]:


sns.catplot(kind='box',x='Age',data=diabetesdata)


# In[126]:


Q1=diabetesdata.Age.quantile(0.25)
Q3=diabetesdata.Age.quantile(0.75)
IQR=Q3-Q1
IQR,Q1,Q3


# # SKIN THICKNESS OUTLIER REMOVAL

# In[127]:


sns.catplot(kind='box',x='SkinThickness',data=diabetesdata)


# In[128]:


Q1=diabetesdata.SkinThickness.quantile(0.25)
Q3=diabetesdata.SkinThickness.quantile(0.75)
IQR=Q3-Q1
IQR,Q1,Q3


# In[129]:


upper_limit=Q3+1.5*IQR
lower_limit=Q1-1.5*IQR
upper_limit,lower_limit


# In[130]:


diabetesdata.SkinThickness=np.where(diabetesdata.SkinThickness>upper_limit,upper_limit,diabetesdata.SkinThickness)


# In[131]:


sns.catplot(kind='box',x='SkinThickness',data=diabetesdata)


# # BMI OUTLIER REMOVAL

# In[132]:


sns.catplot(kind='box',x='BMI',data=diabetesdata)


# In[133]:


Q1=diabetesdata.BMI.quantile(0.25)
Q3=diabetesdata.BMI.quantile(0.75)
IQR=Q3-Q1
IQR,Q1,Q3


# In[134]:


upper_limit=Q3+1.5*IQR
lower_limit=Q1-1.5*IQR
upper_limit,lower_limit


# In[135]:


diabetesdata.BMI=np.where(diabetesdata.BMI>upper_limit,upper_limit,diabetesdata.BMI)
diabetesdata.BMI=np.where(diabetesdata.BMI<lower_limit,lower_limit,diabetesdata.BMI)


# In[136]:


sns.catplot(kind='box',x='BMI',data=diabetesdata)


# # DiabetesPedigreeFunction OUTLIER REMOVAL

# In[137]:


sns.catplot(kind='box',x='DiabetesPedigreeFunction',data=diabetesdata)


# In[138]:


Q1=diabetesdata.DiabetesPedigreeFunction.quantile(0.25)
Q3=diabetesdata.DiabetesPedigreeFunction.quantile(0.75)
IQR=Q3-Q1
IQR,Q1,Q3


# In[139]:


upper_limit=Q3+1.5*IQR
lower_limit=Q1-1.5*IQR
upper_limit,lower_limit


# In[140]:


diabetesdata.DiabetesPedigreeFunction=np.where(diabetesdata.DiabetesPedigreeFunction>upper_limit,upper_limit,diabetesdata.DiabetesPedigreeFunction)


# In[141]:


sns.catplot(kind='box',x='DiabetesPedigreeFunction',data=diabetesdata)


# # Outcome OUTLIER REMOVAL

# In[142]:


diabetesdata.Outcome.unique()


# In[143]:


diabetesdata[diabetesdata.Outcome>1]


# In[144]:


diabetesdata.Pregnancies.unique()


# In[ ]:





# In[145]:


sns.pairplot(diabetesdata)


# # FEATURE ENGINEERING

# In[146]:


X=diabetesdata.drop('Outcome',axis=1)


# In[147]:


# X=X.astype(int)
X


# In[148]:


Y=diabetesdata['Outcome']


# In[149]:


Y


# In[150]:


from sklearn.model_selection import train_test_split


# In[151]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[152]:


print(X.shape,X_test.shape,X_train.shape)


# In[153]:


from sklearn.linear_model import LogisticRegression


# In[154]:


model=LogisticRegression()


# In[155]:


model.fit(X_train,Y_train)


# In[156]:


preds=model.predict(X_train)
print(preds)


# In[157]:


from sklearn.metrics import accuracy_score
trainaccuracy=accuracy_score(Y_train,preds)
trainaccuracy


# In[158]:


preds=model.predict(X_test)


# In[159]:


preds


# In[160]:


testaccuracy=accuracy_score(Y_test,preds)
testaccuracy


# In[163]:


# sample1=[6,148,72,35,0,33.6,0.627,50]
sample1=[5,116,74,0,0,25.6,0.201,30]
finalpred=model.predict(np.array(sample1).reshape(1,-1))
if finalpred==1:
    print ('You have diabetes')
else:
    print ('No diabetes')


# In[ ]:


import pickle as pk


# In[ ]:


model ="diabetespredictionmodel.pkl"
pk.dump(model,open(model,"wb"))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




