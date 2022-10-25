#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("data.csv")


# In[3]:


print(df.head())


# In[4]:


#Grouping all the rows by genre.


print(df.groupby('genre').count())


# In[5]:


print(df.isnull().sum())


# In[ ]:


x=df['summary']
y=df['genre']


# In[6]:


import nltk


# In[7]:


nltk.download("all")


# In[8]:


from nltk.corpus import stopwords


# In[9]:


import re
def clean(x):
  
    x = re.sub("\'", "", x) 
    
    x = re.sub("[^a-zA-Z]"," ",x) 

    x= ' '.join(x.split()) 
    
    x = x.lower() 
    
    return x

df.loc[:,'summary']=df.loc[:,'summary'].apply(lambda x: clean(x))


# In[10]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def removestopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

df['summary'] = df['summary'].apply(lambda x: removestopwords(x))


# In[11]:


print(df['summary'])


# In[12]:


from nltk.stem import WordNetLemmatizer


# In[13]:


l=WordNetLemmatizer()
def lemmatizes(text):
  lem=[l.lemmatize(w) for w in text.split()]
  return ' '.join(lem)
df['summary'] = df['summary'].apply(lambda x: lemmatizes(x))


# In[14]:


print(df['summary'])


# In[15]:


x=df['summary']
y=df['genre']


# In[16]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=5) 


# In[17]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[18]:


t=TfidfVectorizer()
x_train=t.fit_transform(xtrain)
x_test=t.transform(xtest)


# In[ ]:





# In[ ]:



# In[ ]:





# In[ ]:



# In[ ]:




# In[19]:


from sklearn.svm import SVC


# In[20]:


g=SVC(kernel='linear',random_state=0)
g.fit(x_train,ytrain)


# In[21]:


y_p_3=g.predict(x_test)


# In[22]:


from sklearn import metrics


# In[23]:


print(metrics.accuracy_score(ytest,y_p_3))


# In[ ]:

import joblib
# joblib.dump(g,'models/prediction.pkl')
joblib.dump(t,'models/transform.pkl')


# In[ ]:




# In[ ]:





# In[24]:




# In[25]:


import wikipedia


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




# In[ ]:





# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



# In[ ]:




# In[ ]:




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




# In[26]:





# In[31]:


from gtts import gTTS




my_text = input(' Enter text-to-speech:\n > ')
r=wikipedia.page(my_text).content
p=[]
q=[]
p.append(my_text)
q.append(r)
df3=pd.DataFrame({'title':p,'summary':q})
df3.loc[:,'summary']=df3.loc[:,'summary'].apply(lambda x: clean(x))
df3['summary'] = df3['summary'].apply(lambda x: removestopwords(x))
df3['summary'] = df3['summary'].apply(lambda x: lemmatizes(x))
x8=df3['summary']
x8=t.transform(x8)
y8=g.predict(x8)
k=y8[0]
m="The category of book is "+k
tts = gTTS(text=m, lang='en', slow=False)
tts.save('summarys.mp3')  



print("\n  Done! ")



input('\n'*2 + '  Hit enter to exit...')


# In[ ]:




# In[32]:


import IPython
IPython.display.Audio('summarys.mp3')

