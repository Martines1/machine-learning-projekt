import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, \
    precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import models

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
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.6
)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)


#Treba vybrat metodu z models.py
#selected_method = models.find_SVC_model(X_train_vec, y_train)
selected_method, tag = models.find_LR_model(X_train_vec, y_train)
y_train_pred = selected_method.predict(X_train_vec)

train_accuracy = accuracy_score(y_train, y_train_pred)
train_report = classification_report(y_train, y_train_pred)
print("\nTraining Classification Report:")
print(train_report)

print("Training Accuracy:", train_accuracy)

# Spustim model na validacnych a testovacich datach
y_val_pred = selected_method.predict(X_val_vec)
y_test_pred = selected_method.predict(X_test_vec)

# Ziskam report, accuracy a confusion matrix
val_report = classification_report(y_val, y_val_pred)
test_report = classification_report(y_test, y_test_pred)

val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

confusion_mat = confusion_matrix(y_test, y_test_pred)

# Printovanie jednotlivych vysledkov
print("Distribution of training/validating/testing data:")
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

print(f"Accuracy: {np.round(accuracy, 3)}")
print(f"Macro Precision: {np.round(precision,3)}")
print(f"Macro Recall: {np.round(recall, 3)}")
print(f"Macro F1-Score: {np.round(f1, 3)}")

#ukazka confusion matrix
disp = ConfusionMatrixDisplay(confusion_mat, display_labels=['Action', 'Comedy', 'Horror'])
disp.plot(cmap='Blues')
plt.title(tag + " Confusion Matrix for test Data")
plt.show()