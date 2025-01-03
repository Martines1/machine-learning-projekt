import pandas as pd
from langid import langid

import text_preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle

#Vybrate zanre
selected_genres = ["Comedy",  "Horror", "Action"]
file_path = 'TMDB_all_movies.csv'
columns = ['id', 'title', 'overview', 'genres']
df = pd.read_csv(file_path, usecols=columns, keep_default_na=False)
# Ak sa prevod nepodari, nastavi hodnotu na NaN
df['id'] = pd.to_numeric(df['id'], errors='coerce')
# Odstranim riadky, kde je 'id' NaN
df = df.dropna(subset=['id'])
df['id'] = df['id'].astype(int)
#odstranim prazdne hodnoty
df['overview'] = df['overview'].fillna('').str.strip()
df['genres'] = df['genres'].fillna('').str.strip()
df = df[(df['overview'] != '') & (df['genres'] != '')]
# Odstranenim duplicit na zaklade overview
df = df.drop_duplicates(subset=['overview'])

#Skontrolujem ci je overview v anglictine
def is_english(text):
    try:
        lang, confidence = langid.classify(text)
        return lang == "en"
    except:
        return False
#ak nie tak ho odstranim
df["is_english"] = df["overview"].apply(is_english)
df= df[df["is_english"] == True]

df['genres_list'] = df['genres'].str.split(', ')
for genre in selected_genres:
    df[genre] = df['genres_list'].apply(lambda x: 1 if isinstance(x, list) and genre in x else 0)

# Odfiltrovanie filmov, ktore nemaju ziaden z vybranych zanrov alebo viacej ako jeden
df = df[df[selected_genres].sum(axis=1) == 1]


def assign_target_genre(genres_list, selected_genres):
    if isinstance(genres_list, list):
        for genre in selected_genres:
            if genre in genres_list:
                return genre
    return None

df['target_genre'] = df['genres_list'].apply(lambda x: assign_target_genre(x, selected_genres))

#preprocessing 2
df['overview'] = df['overview'].apply(text_preprocessing.remove_interpunction)
df['overview'] = df['overview'].apply(text_preprocessing.remove_non_ascii)
df['overview'] = df['overview'].apply(text_preprocessing.remove_tags)
df['overview'] = df['overview'].apply(text_preprocessing.remove_digits)
df['overview'] = df['overview'].apply(text_preprocessing.keep_alphabet_only)
df['overview'] = df['overview'].apply(text_preprocessing.remove_stopWords)

# df['overview'] = df['overview'].apply(text_preprocessing.apply_stemmer)
df['overview'] = df['overview'].apply(text_preprocessing.apply_lemmantizer)
df = df.drop_duplicates(subset=['overview'])
df = df[df['overview'] != '']
df['word_count'] = df['overview'].apply(lambda x: len(x.split()))
df = df[df['word_count'] > 2]
df = df.drop(columns=['word_count'])


df['overview_word_count'] = df['overview'].fillna('').apply(lambda x: len(x.split()))

#zhodnotim kvalitu overview a vyberiem top k
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['overview'])
tfidf_sum = tfidf_matrix.sum(axis=1)
tfidf_sum = tfidf_sum.A1
df['tfidf_sum'] = tfidf_sum
df = df.sort_values(by='tfidf_sum', ascending=False)
threshold = df['tfidf_sum'].quantile(0.7)
print(threshold)

df = df[df['tfidf_sum'] >= threshold].reset_index(drop=True)

print(df[['overview', 'tfidf_sum']])
max_per_genre = 10000
balanced_dfs = []
genre_counts = {genre: 0 for genre in selected_genres}
for _, row in df.iterrows():
    film_genres = [genre for genre in selected_genres if row[genre] == 1]

    if all(genre_counts[genre] < max_per_genre for genre in film_genres):
        balanced_dfs.append(row)
        for genre in film_genres:
            genre_counts[genre] += 1

df_balanced = pd.DataFrame(balanced_dfs)
duplicate_count = df_balanced['overview'].duplicated().sum()
print(f"Pocet duplicit v 'overview': {duplicate_count}")
genre_counts = {genre: df_balanced[df_balanced[genre] == 1].shape[0] for genre in selected_genres}
for genre, count in genre_counts.items():
    print(f"{genre}: {count} filmov")
df_balanced = shuffle(df_balanced, random_state=42)
df_balanced.to_csv("movies_clean_copy.csv", index=False)