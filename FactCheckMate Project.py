#!/usr/bin/env python
# coding: utf-8

# # FactCheckMate

# In[14]:


# python project on detection of fake news using ML


# In[15]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# In[16]:


# use fake new and real news dataset
df_fake=pd.read_csv("Fake.csv")
df_true=pd.read_csv("True.csv")


# In[17]:


df_fake.head(10)


# In[18]:


df_true.head(10)


# In[19]:


df_fake.shape


# In[20]:


df_true.shape


# In[21]:


# Inserting a column called "class" for fake and real news dataset to categories fake and true news.
df_fake["class"] = 0
df_true["class"] = 1


# In[22]:


df_fake_test=df_fake.tail(10)


# In[23]:


for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)
df_true_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)


# In[24]:


# creating a manual testing csv for testing at the end
df_manual_testing=pd.concat([df_fake_test,df_true_testing],axis=0)
df_manual_testing


# In[25]:


df_manual_testing.to_csv("manual_testing")


# In[26]:


# creating a single dataframe for fake and true news so that we can use it for training purpose
df_training=pd.concat([df_fake,df_true],axis=0)


# In[27]:


# title column , subject and date these 3 columns are not useful for our detection system so we will remove them
df=df_training.drop(["title","subject","date"],axis=1)


# In[28]:


# to shuffle our dataset
df=df.sample(frac=1)


# In[29]:


df.head(5)


# In[30]:


#checking presence of null values
df.isnull().sum()


# In[31]:


# removing all unuseful text from our dataset
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

df["text"] = df["text"].apply(wordopt)


# In[32]:


# creating x and y for training dataset
x=df["text"]
y=df["class"]


# In[33]:


#Splitting the dataset into training set and testing set. 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# In[34]:


# convert text to vectors
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# In[35]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression


# In[36]:


LR = LogisticRegression()
LR.fit(xv_train,y_train)
pred_lr=LR.predict(xv_test)


# In[37]:


LR.score(xv_test, y_test)


# In[38]:


print(classification_report(y_test, pred_lr))


# In[39]:


# decision tree
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
DT.score(xv_test, y_test)


# In[40]:


print(classification_report(y_test, pred_dt))


# In[41]:


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)
pred_rfc = RFC.predict(xv_test)
RFC.score(xv_test, y_test)


# In[42]:


print(classification_report(y_test, pred_rfc))


# In[44]:


# Model testing with user input of news
import tkinter as tk
from tkinter import messagebox
import pandas as pd

# Define the functions
def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "True News"

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)  # Assuming wordopt function is defined elsewhere
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)  # Assuming 'vectorization' is defined elsewhere
    pred_LR = LR.predict(new_xv_test)  # Assuming LR, DT, GBC, RFC are defined elsewhere
    pred_DT = DT.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    result_str = "\n\nLR Prediction: {} \nDT Prediction: {} \nRFC Prediction: {}".format(
        output_label(pred_LR[0]), output_label(pred_DT[0]), output_label(pred_RFC[0]))
    
    messagebox.showinfo("Prediction Results", result_str)

def on_submit():
    news = text_input.get("1.0", "end-1c")
    manual_testing(news)

# Create the main window
root = tk.Tk()
root.title("Fake News Detection")

# Create text input widget
text_input = tk.Text(root, height=10, width=50)
text_input.pack()

# Create submit button
submit_button = tk.Button(root, text="Submit", command=on_submit)
submit_button.pack()

# Run the main event loop
root.mainloop()


# In[ ]:




