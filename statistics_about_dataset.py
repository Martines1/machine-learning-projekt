import pandas as pd
import matplotlib.pyplot as plt


file_path = 'TMDB_all_movies.csv'
columns = ['id', 'title', 'overview', 'genres']
df = pd.read_csv(file_path, usecols=columns, keep_default_na=False)

df['genres_list'] = df['genres'].str.split(', ')

selected_genres = ["Comedy", "Horror", "Action", "Romance", "Thriller", "Drama", "Science Fiction", "Fantasy", "Animation", "Documentary", "History", "Crime", "War", "Western"]


genre_counts = {}

for genres in df['genres_list']:
    for genre in genres:
        if genre in selected_genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        else:
            genre_counts['Others'] = genre_counts.get('Others', 0) + 1
            print(genre)

plt.figure(figsize=(10, 6))
plt.bar(genre_counts.keys(), genre_counts.values())
plt.xlabel('Genre')
plt.ylabel('Number of movies')
plt.title('Number of movies by genre')
total_rows = len(df)
print(f"Pocet riadkov: {total_rows}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Statistika
empty_overview_count = df['overview'].apply(lambda x: x.strip() == '').sum()
empty_genres_count = df['genres'].apply(lambda x: x.strip() == '').sum()



filtered_overview = df['overview'].apply(lambda x: x.strip() != '')
duplicate_overview_count = df.loc[filtered_overview, 'overview'].duplicated().sum()


print(f"Pocet riadkov s prazdnym alebo NaN overview: {empty_overview_count}")
print(f"Pocet riadkov s prazdnymi alebo NaN genres: {empty_genres_count}")
print(f"Počet duplicitnych overview: {duplicate_overview_count}")
