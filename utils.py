import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)

sns.set_style('darkgrid')

def drop_outliers(df,col):
    q25,q75=np.percentile(df[col],25),np.percentile(df[col],75)
    iqr=q75-q25
    cut_off=iqr*1.5

    lower_lim=q25-cut_off
    uper_lim=q75+cut_off

    outliers=[x for x in df[col] if x>uper_lim or x<lower_lim]

    print("outliers identified for column {} are:{}\n".format(col,len(outliers)))
    
    df=df[~df[col].isin(outliers)]
    
    return df

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

stops=stopwords.words("english")

def preprocess_txt(data):
    stops=stopwords.words("english")

    #data['STORY']=data.STORY.apply(lambda x:contractions.fix(x.lower()))
    data=data.apply(lambda x:word_tokenize(x.lower()))
    stops=stopwords.words('english')
    data=data.apply(lambda x:[i for i in x if i.isalpha()])
    data=data.apply(lambda x:[i for i in x if i not in stops])
    data=data.apply(lambda x:[WordNetLemmatizer().lemmatize(i) for i in x])
    data=data.apply(lambda x:' '.join([str(i) for i in x]))
    
    return data



def DFboxplot(df):
    fig = plt.figure(figsize=(10, 10))
    x=df.select_dtypes('number')
    print("columns={}\n".format(x.columns))
    n=len(list(x.columns))
    pn = 1
    for col in x:
        print("n={}\npn={}\n".format(n,pn))
        if pn <= n:
            plt.subplot(3, np.round(n / 3), pn)
            g = sns.boxplot(x[col])
            plt.xlabel(col, fontsize=15,color='red')
        pn += 1
    plt.show()
    
    
def DFdistplot(df):
    fig = plt.figure(figsize=(10, 10))
    x=df.select_dtypes('number')
    n=len(list(x.columns))
    pn = 1
    for col in x.columns:
        if pn <= n:
            plt.subplot(3, np.round(n / 3), pn)
            g = sns.distplot(x[col])
            plt.xlabel(col, fontsize=15,color='red')
        pn += 1
    plt.show()