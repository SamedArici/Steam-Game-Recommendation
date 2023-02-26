import numpy as np
import pandas as pd

import math

import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# adding the datasets
steam = pd.read_csv('./datasets/steam.csv')
df_steam = steam.copy()
df_desc = pd.read_csv('./datasets/steam_description_data.csv')
df_media = pd.read_csv('./datasets/steam_media_data.csv')
df_tags = pd.read_csv('./datasets/steamspy_tag_data.csv')

# Editing screenshot column in df_media
def screenshots(ss):
    im = ''
    ss = ss.split(',')
    for i in range(2,len(ss),3):
        ss[i] = ss[i].replace('}','').replace(']','')
        im += ss[i].split(' ')[2].replace("'","") + ','
    
    return im

df_media['screenshots'] = df_media['screenshots'].apply(screenshots)

# creating a new dataframe
df = df_steam.merge(df_desc,left_on = 'appid',right_on = 'steam_appid')
df = df.merge(df_media,left_on = 'appid',right_on = 'steam_appid')
df_new_tags = df_steam.merge(df_tags,on = 'appid')

tags = []
cols = list(df_tags.columns)
cols.remove('appid')
l = list(df_steam.appid.unique())
for i in df_new_tags[df_tags.columns].values:
    t = []
    for index,value in enumerate(i[1:],start=0):
        if value > 0:
            t.append(cols[index])     
    tags.append(' '.join(t))
    
    
# Editing the columns
df['tags'] = np.array(tags)

df['year'] = df['release_date'].apply(lambda x: x.split('-')[0])

df['categories'] = df['categories'].apply(lambda x: x.replace(';',' '))
df['developer'] = df['developer'].apply(lambda x: x.replace(' ','').replace(',',' ').replace(";",' ').replace('-','').replace(' /',' '))
df['publisher'] = df['publisher'].apply(lambda x: x.replace(' ','').replace(',',' ').replace(";",' ').replace('-','').replace(' /',' '))
df['genres'] = df['genres'].apply(lambda x: x.replace(';',' '))


def age(x):
    if x == 0:
        return 'allAge'
    elif x < 12:
        return 'child'
    elif x < 18:
        return 'teenager'
    
    return 'adult'

df['required_age'] = df['required_age'].apply(age)


# Calculate Review Score
total_reviews = df['positive_ratings'] + df['negative_ratings']
review_score = df['positive_ratings'] / total_reviews
score_cat = []
for i in range(0, len(review_score)):
    
    score = review_score[i] *100
    review = total_reviews[i]
    
    if score >= 95:
        if review >=500:
            score_cat.append('Overwhelmingly Positive')
        elif review >= 50:
            score_cat.append('Very positive')
        else:
            score_cat.append('Positive')
            
    elif score >= 80:
        if review >=50:
            score_cat.append('Very Positive')
        else:
            score_cat.append('Positive')
            
    elif score >= 70:
        score_cat.append('Mostly Positive')
    
    elif score >= 40:
        score_cat.append('Mixed')
        
    elif score >= 20:
        score_cat.append('Mostly Negative')
        
    else:
        if review >=500:
            score_cat.append('Overwhelmingly Negative')
        
        elif review >= 50:
            score_cat.append('Very Negative')
            
        else:
            score_cat.append('Negative')
df['review_score'] = np.array(score_cat)


# Calculate Rate of the game
for i in range(0,len(total_reviews)):
    total_reviews[i] = math.pow(2,-math.log10(total_reviews[i] + 1))
    
rate = review_score - (review_score-0.5) * total_reviews
df['rate'] = 100*round(rate,2)
df['rate'] = df['rate'].astype('int')


# Create the result dataframe
df_final = df[[
    'name',
    'header_image',
    'short_description',
    'release_date',
    
    'required_age', 
    'categories',
    'tags',
    'genres',
    'review_score',
    'developer',
    'publisher',
    'rate',
    'year'
    
]].copy()

def clear_tags(x):
    x = x.lower()
    x = x.replace('_','').replace('-','')
    x = x.split(' ')
    x = list(dict.fromkeys(x))
    return ' '.join(x) 

for i in list(df_final.select_dtypes(['float','int']).columns):
    df_final[i] = df_final[i].astype('str')

df_final['all'] = df_final['categories'] + ' '+ df_final['genres'] + ' '+ df_final['tags']
df_final['all'] = df_final['all'].apply(clear_tags)

nltk.download('stopwords')
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
ps = PorterStemmer()


df_final['text'] = ''
text_cols = ['all','developer','publisher','review_score','required_age','year','rate']
for c in text_cols:
    df_final['text'] += ' '+ df_final[c]


corpus = []
for s in range(len(df_final)):
    text = re.sub('[^a-zA-Z0-9]',' ',df_final['text'][s])
    text = text.lower().split()
    text = [ps.stem(word) for word in text if not word in set(all_stopwords)]
    text = ' '.join(text)
    corpus.append(text)


cv = CountVectorizer(max_features=2000)
vector = cv.fit_transform(corpus).toarray()
similarity = cosine_similarity(vector)

# editing game names
df.name = df.name.apply(lambda x: re.sub('[®™]','',x).lower())
def roman_to_int(s):
    try:
        roman = {'i':1,'v':5,'x':10,'l':50,'c':100,'d':500,'m':1000,'iv':4,'ix':9,'xl':40,'xc':90,'cd':400,'cm':900}
        i = 0
        num = 0
        while i < len(s):
            if i+1<len(s) and s[i:i+2] in roman:
                num+=roman[s[i:i+2]]
                i+=2
            else:
                num+=roman[s[i]]
                i+=1
        return str(num)
    except:
        return s
def edit_name(i):
    i = i.split(' ')
    for s in range(0,len(i)):
        i[s] = roman_to_int(i[s])
   
    return ' '.join(i)
df['name'] = df['name'].apply(edit_name)

steam.name = df.name.copy()
steam.developer = steam.developer.apply(lambda x: x.replace(';',', ').replace(' /',','))
steam.publisher = steam.developer.apply(lambda x: x.replace(';',', ').replace(' /',','))
steam.release_date = steam.release_date.apply(lambda x: f"{x.split('-')[2]}.{x.split('-')[1]}.{x.split('-')[0]}")
steam['review_score'] = df.review_score.copy()
steam['short_description'] = df.short_description.copy()
steam['header_image'] = df.header_image.copy()
steam['screenshots'] = df.screenshots.copy()
steam[['name','review_score','release_date','developer','publisher','short_description','header_image','screenshots']].to_csv('dataframe.csv',index=False)

with open('similarity.npy', 'wb') as f:
    np.save(f, similarity)