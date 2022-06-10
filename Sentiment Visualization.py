#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# pip install contractions


# In[ ]:


# pip install -U textblob


# In[ ]:


# pip install scattertext


# In[91]:


# import the package needed
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import glob
import re, string, unicodedata
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
import contractions
from textblob import TextBlob
##import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize import word_tokenize,RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.stem import PorterStemmer
##import viz packages
import itertools
import collections
import matplotlib.pyplot as plt
import scattertext as st
from IPython.display import IFrame
from IPython.core.display import display, HTML
from scattertext import CorpusFromPandas, produce_scattertext_explorer


# In[10]:


path = r'D:\Project\Movie_reviews_labeled' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

Reviews = pd.concat(li, axis=0, ignore_index=True)


# In[11]:


path = r'D:\Project\Critic_Reviews_labeled' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, lineterminator='\n', index_col=None, header=0)
    li.append(df)

Critics = pd.concat(li, axis=0, ignore_index=True)


# In[12]:


IMDB = Reviews.iloc[:,2:4]
IMDB.head()


# In[13]:


Metacritics = Critics.iloc[:,6:8]
Metacritics.head()


# In[14]:


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

def preprocess(Text):
    Text=str(Text)
       
    # Lowering the text
    Text = Text.lower()
    
    # Removing Weblinks
    Text=Text.replace('{html}',"")
    
    # Removing Special characters
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', Text)
    
    #Removing URLs
    rem_url=re.sub(r'http\S+', '',cleantext)
    
    # Removing Numbers
    rem_num = re.sub('[0-9]+', '', rem_url)
    
    # Lemmatization
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    
    # Removing Stopwords.
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    
    return " ".join(filtered_words)


# In[15]:


IMDB['Comment_cleaned']=IMDB['Comment'].map(lambda s:preprocess(s)) 
IMDB.head()


# In[16]:


Metacritics['Review_cleaned']=Metacritics['Review'].map(lambda s:preprocess(s)) 
Metacritics.head()


# In[17]:


#word lemmatization for IMDB
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
clean_lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_cleantext(text):
    return [clean_lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


# In[20]:


#word lemmatization for IMDB
IMDB['Comment_lem'] = IMDB['Comment_cleaned'].apply(lemmatize_cleantext)
IMDB.head()


# In[21]:


#plotting most common words in IMDB reviews
IMDB_cleaned_comments = list(IMDB['Comment_lem'])
IMDB_comment_list = list(itertools.chain(*IMDB_cleaned_comments))
counts_no = collections.Counter(IMDB_comment_list)
IMDB_clean_comments = pd.DataFrame(counts_no.most_common(30),columns=['words','count'])
fig, ax = plt.subplots(figsize=(12,8))
IMDB_clean_comments.sort_values(by='count').plot.barh(x='words',y='count',ax=ax, color='blue')
ax.set_title("Most frequently used words in IMDB User Reviews")
plt.show()


# In[22]:


#word lemmatization for Metacritics
Metacritics['Review_lem'] = Metacritics['Review_cleaned'].apply(lemmatize_cleantext)
Metacritics.head()


# In[23]:


#plotting most common words in metacritic reviews
Metacritics_cleaned_reviews = list(Metacritics['Review_lem'])
Metacritics_review_list = list(itertools.chain(*Metacritics_cleaned_reviews))
counts_no = collections.Counter(Metacritics_review_list)
Metacritics_clean_revs = pd.DataFrame(counts_no.most_common(30),columns=['words','count'])
fig, ax = plt.subplots(figsize=(12,8))
Metacritics_clean_revs.sort_values(by='count').plot.barh(x='words',y='count',ax=ax, color='blue')
ax.set_title("Most frequently used words in Critic Reviews")
plt.show()


# In[24]:


#Parsed IMDB df for scattertext visualization - not adding into main IMDB database as content too redundant, creating new df
IMDB_parsed = IMDB.assign(Comment_parse=lambda df: IMDB.Comment_cleaned.apply(st.whitespace_nlp_with_sentences))
IMDB_parsed.head()


# In[25]:


#IMDB scattertext plot
corpus = st.CorpusFromParsedDocuments(IMDB_parsed,
                                      category_col='Emotion',
                                      parsed_col='Comment_parse').build().get_unigram_corpus().compact(st.AssociationCompactor(2000))
html = st.produce_scattertext_explorer(corpus,
                                      category='positive',
                                      category_name='positive',
                                      not_category_name='negative',
                                      minimum_term_frequency=5,
                                      width_in_pixels=1000,
                                      transform=st.Scalers.log_scale_standardize)
open('IMDB Sentiment Visualization.html', 'wb').write(html.encode('utf-8'))


# In[26]:


#Parsed Metacritics df for scattertext
Metacritics_parsed = Metacritics.assign(Review_parse=lambda df: Metacritics.Review_cleaned.apply(st.whitespace_nlp_with_sentences))
Metacritics_parsed.head()


# In[27]:


#Metacritics scattertext plot
corpus = st.CorpusFromParsedDocuments(Metacritics_parsed,
                                      category_col='Emotion',
                                      parsed_col='Review_parse').build().get_unigram_corpus().compact(st.AssociationCompactor(2000))
html = st.produce_scattertext_explorer(corpus,
                                      category='positive',
                                      category_name='positive',
                                      not_category_name='negative',
                                      minimum_term_frequency=5,
                                      width_in_pixels=1000,
                                      transform=st.Scalers.log_scale_standardize)
open('Metacritics Sentiment Visualization.html', 'wb').write(html.encode('utf-8'))


# In[42]:


def sigmoid(z): 
    # calculate the sigmoid of z
    h = 1/(1 + np.exp(-z))    
    return h


# In[54]:


def gradientDescent(x, y, theta, alpha, num_iters):
    #theta: weight vector
    #alpha: learning rate
    #num_iters: number of iterations for training
    #output:
    #J: the final cost
    #theta: your final weight vector
    
    m = x.shape[0]
  
    for i in range(0, num_iters):
        
        # get z, the dot product of x and theta
        z = np.dot(x,theta)
        
        # get the sigmoid of z
        h = sigmoid(z)
        
        # calculate the cost function
        J = -(np.dot(y.T,np.log(h))+np.dot((1-y).T,np.log(1-h)))/m
        
        # update the weights theta
        theta = theta - alpha*(np.dot(x.T,h-y))/m
        
    J = float(J)
    return J, theta


# In[116]:


#train test split
X = IMDB['Comment_cleaned']
#binarizing Y before split
lb=LabelBinarizer()
y = IMDB['Emotion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)


# In[106]:


#Bag of Words 
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
#transformed train x
cv_train_IMDB=cv.fit_transform(X_train)
#transformed test x
cv_test_IMDB=cv.transform(X_test)

print('BOW_cv_train_IMDB:',cv_train_IMDB.shape)
print('BOW_cv_test_IMDB:',cv_test_IMDB.shape)
#vocab=cv.get_feature_names()-toget feature names


# In[107]:


#TF-IDF
tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
#transformed train x
tv_train_IMDB=tv.fit_transform(X_train)
#transformed test x
tv_test_IMDB=tv.transform(X_test)

print('Tfidf_train_IMDB:',tv_train_IMDB.shape)
print('Tfidf_test_IMDB:',tv_test_IMDB.shape)


# In[108]:


#training the model
lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)

#Fitting the model for Bag of words
lr_bw=lr.fit(cv_train_IMDB,y_train)
print(lr_bw)


# In[109]:


#Fitting the model for tfidf features
lr_tfidf=lr.fit(tv_train_IMDB,y_train)
print(lr_tfidf)


# In[110]:


#Predicting the model for bag of words
lr_bw_predict=lr.predict(cv_test_IMDB)
print(lr_bw_predict)


# In[111]:


##Predicting the model for tfidf features
lr_tfidf_predict=lr.predict(tv_test_IMDB)
print(lr_tfidf_predict)


# In[112]:


#Accuracy score for bag of words
lr_bw_score=accuracy_score(y_test,lr_bw_predict)
print("Bag of Words Score:",lr_bw_score)

#Accuracy score for tfidf features
lr_tfidf_score=accuracy_score(y_test,lr_tfidf_predict)
print("TF-IDF Score:",lr_tfidf_score)


# In[113]:


#Classification report for bag of words 
lr_bw_report=classification_report(y_test,lr_bw_predict,target_names=['Positive','Negative'])
print(lr_bw_report)
#confusion matrix for bag of words
cm_bow=confusion_matrix(y_test,lr_bw_predict,labels=['positive','negative'])
print(cm_bow)


# In[114]:


#Classification report for tfidf features
lr_tfidf_report=classification_report(y_test,lr_tfidf_predict,target_names=['Positive','Negative'])
print(lr_tfidf_report)
#confusion matrix for tfidf features
cm_tfidf=confusion_matrix(y_test,lr_tfidf_predict,labels=['positive','negative'])
print(cm_tfidf)


# In[ ]:




