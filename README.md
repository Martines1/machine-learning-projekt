# machine-learning-projekt
Projekt na predmet Strojové učenie - téma sa volá Klasifikácia filmových žánrov \
Postup: \
Je potrebna Anaconda, aby bolo možné nainštalovať prostredie pomocou
```
conda env create -f environment.yml
```
Následne sa kód spúšta cez toto prostredie.
Popis súborov\
evaluate.py - slúži na vyhodnotenie jednotlivých metód v models.py \
prepare_movies.py - pripraví dataset na trenovanie. Je potrebné mať stiahnutý dataset z https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies \
movies_clean.csv - pripravený dataset na vyhodnotenie \
text_preprocessing - NLP príprava textu na vstup pre model \

