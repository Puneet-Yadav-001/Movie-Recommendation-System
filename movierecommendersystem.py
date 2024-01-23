import pandas as pd
import numpy as np

###Importing Datasets

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies.head(1)

credits.head(1)

movies.shape

credits.shape

###Merging both datasets

movies = movies.merge(credits, on="title")

movies.head(1)

movies.shape

###Removing unwanted columns from datasets

movies.info()

# genres
# id
# keywords
# title
# overview
# cast
# crew

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

movies.head(1)

###Preprocessing on dataset

movies.isnull().sum()

movies.duplicated().sum()

movies.dropna(inplace=True)

movies.isnull().sum()

movies.iloc[0]['genres']

import ast

def convert(obj):
  l = []
  for i in ast.literal_eval(obj):
    l.append(i["name"])
  return l

movies['genres'] = movies['genres'].apply(convert)

movies.head(1)

movies['keywords'] = movies['keywords'].apply(convert)

movies.head(1)

def getName(obj):
  l = []
  for i in ast.literal_eval(obj):
    if i['order'] < 3:
      l.append(i['name'])
  return l

movies['cast'] = movies['cast'].apply(getName)

movies.head(1)

movies['crew'][0]

def getName2(obj):
  l = []
  for i in ast.literal_eval(obj):
    if i['job'] == "Director":
      l.append(i['name'])
      break
  return l

movies['crew'] = movies['crew'].apply(getName2)

movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies.head()

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(' ', '') for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(' ', '') for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(' ', '') for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(' ', '') for i in x])

movies.head()

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

movies.head()

new_df = movies[['movie_id', 'title', 'tags']]

new_df

new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))

new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

new_df['tags'][0]

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
  y = []

  for i in text.split():
    y.append(ps.stem(i))

  return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

###Vectorization

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

vectors

cv.get_feature_names_out

###Cosine Similarity

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

similarity

def recommend(movie):
  movie_index = new_df[new_df['title'] == movie].index[0]
  distances = similarity[movie_index]
  movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
  for i in movies_list:
    print(new_df.iloc[i[0]].title)

recommend("Escape from Planet Earth")