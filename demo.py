import pyttsx3
import pandas as pd
from gtts import gTTS
import wikipedia
import joblib
import playsound
g=joblib.load('models/prediction.pkl')
t=joblib.load('models/transform.pkl')
import nltk
import re
engine=pyttsx3.init()
voice=engine.getProperty('voices')
engine.setProperty('voice',voice[1].id)
engine.setProperty("rate",130)
def clean(x):
  
    x = re.sub("\'", "", x) 
    
    x = re.sub("[^a-zA-Z]"," ",x) 

    x= ' '.join(x.split()) 
    
    x = x.lower() 
    
    return x
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def removestopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)
from nltk.stem import WordNetLemmatizer
l=WordNetLemmatizer()
def lemmatizes(text):
  lem=[l.lemmatize(w) for w in text.split()]
  return ' '.join(lem)
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
tts=gTTS(m,lang='en')
files="summarys.mp3"
tts.save(files)
playsound.playsound(files)
print(m)