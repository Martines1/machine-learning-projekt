import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support, \
    ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("movies_clean.csv")
X = df['overview']   # vstupne data
y = df['target_genre'] # cielove data

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Rozdelenie dat na 60% trening, 20% validacia, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# TF-IDF vektorizacia
vectorizer = TfidfVectorizer(
    max_features=70000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.6
)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)


def get_DT_model(max_depth):
    dt_model_ = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    dt_model_.fit(X_train_vec, y_train)
    return dt_model_

def find_DT_model():
    dt_param_grid = {'max_depth': [3, 5, 10, 15, 20, 30, 50, 100, 200], 'criterion': ["gini", "entropy"]}
    dt_grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    dt_grid_search.fit(X_train_vec, y_train)
    print("Najlepšie parametre DT:", dt_grid_search.best_params_)
    return dt_grid_search.best_estimator_


def get_RF_model(max_depth, n_estimators, criterion):
    rf_model_ = RandomForestClassifier(random_state=42, n_jobs=-1, criterion=criterion, max_depth=max_depth, n_estimators=n_estimators)
    rf_model_.fit(X_train_vec, y_train)
    return rf_model_

def find_RF_model():
    rf_param_grid = {'max_depth': [3, 5, 10, 15, 20, 30, 50, 100, 200], 'n_estimators': [5, 10, 50, 100, 200, 500], 'criterion': ["gini", "entropy"]}
    rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    rf_grid_search.fit(X_train_vec, y_train)
    print("Najlepšie parametre DT:", rf_grid_search.best_params_)
    return rf_grid_search.best_estimator_

def get_GB_model(learning_rate, max_depth):
    gb_model_ = GradientBoostingClassifier(random_state=42, learning_rate=learning_rate, max_depth=max_depth)
    gb_model_.fit(X_train_vec, y_train)
    return gb_model_

def find_GB_model():
    gb_param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5],
        'max_depth': [3, 10, 15, 20, 30]
    }
    gb_grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_param_grid, cv=5, scoring='accuracy',
                            n_jobs=-1, verbose=3)
    gb_grid_search.fit(X_train_vec, y_train)
    print("Najlepšie parametre DT:", gb_grid_search.best_params_)
    return gb_grid_search.best_estimator_


selected_model = get_GB_model(max_depth=10, learning_rate=0.5)
# Spustim model na validacnych a testovacich datach
y_val_pred = selected_model.predict(X_val_vec)
y_test_pred = selected_model.predict(X_test_vec)

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
plt.title("Confusion Matrix for test Data")
plt.show()
