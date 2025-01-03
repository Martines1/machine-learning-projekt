import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def get_model(kernel, C, gamma):
    svc_model_ = SVC(kernel=kernel, C = C, gamma=gamma, random_state=42, verbose=True, shrinking=False, cache_size=2000)
    svc_model_.fit(X_train_vec, y_train)
    return svc_model_

def find_model():
    param_grid = {
        'C': [0.1, 1, 10, 20, 50],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(
        estimator=SVC(random_state=42, verbose=True, shrinking=False, cache_size=2000),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        verbose=3,
        n_jobs=-1
    )
    grid_search.fit(X_train_vec, y_train)
    print("Najlepšie parametre:", grid_search.best_params_)
    return grid_search.best_params_


df = pd.read_csv("movies_clean.csv")
X = df['overview']   # vstupne data
y = df['target_genre'] # cielove data

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Rozdelim data na trenovacie 60% / testovacie 20% / validacne 20%
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Vektorizovanie dat pomocou TF-IDF
vectorizer = TfidfVectorizer(
    max_features=70000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.6
)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# Ak by som chcel hladat parametre podla grid search
#svc_model = find_model()
# Ak uz mam vopred vybrate parametre
svc_model = get_model(C=10, kernel='rbf', gamma='scale')

# Spustim model na validacnych a testovacich datach
y_val_pred = svc_model.predict(X_val_vec)
y_test_pred = svc_model.predict(X_test_vec)

# Ziskam report, accuracy a confusion matrix
val_report = classification_report(y_val, y_val_pred)
test_report = classification_report(y_test, y_test_pred)

val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

confusion_mat = confusion_matrix(y_test, y_test_pred)

# Printovanie jednotlivych vysledkov
print("Distribucia trenovacich/validacnych/testovacich dat:")
print("Train:", np.bincount(y_train))
print("Validation:", np.bincount(y_val))
print("Test:", np.bincount(y_test))

print("\nValidation Classification Report:")
print(val_report)

print("Validation Accuracy:", val_accuracy)

print("\nTest Classification Report:")
print(test_report)

print("Test Accuracy:", test_accuracy)

print("\nConfusion Matrix (Test):")
print(confusion_mat)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='macro')
accuracy = accuracy_score(y_test, y_test_pred)

print(f"Accuracy: {accuracy}")
print(f"Macro Precision: {precision}")
print(f"Macro Recall: {recall}")
print(f"Macro F1-Score: {f1}")

#ukazka confusion matrix
disp = ConfusionMatrixDisplay(confusion_mat, display_labels=['Action', 'Comedy', 'Horror'])
disp.plot(cmap='Blues')
plt.title("SVC Confusion Matrix for test Data")
plt.show()