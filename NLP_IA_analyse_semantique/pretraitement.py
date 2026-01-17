import pandas as pd
import numpy as np
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Charger les données
# On charge le fichier CSV contenant les critiques de films et leurs sentiments (positif ou négatif)
df = pd.read_csv("IMDB Dataset.csv")
print(df.head())

# 2. Vérification basique
# On vérifie qu'il n'y a pas de valeurs manquantes et on observe la répartition des classes
print(df.info())
print(df['sentiment'].value_counts())

# 3. Nettoyage du texte
# On définit une fonction pour nettoyer chaque review :
# - suppression des balises HTML
# - suppression des caractères spéciaux
# - conversion en minuscules
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # supprimer HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # garder que lettres
    text = text.lower()
    return text

df['review'] = df['review'].apply(clean_text)

# 4. Encodage des labels
# On convertit les labels "positive"/"negative" en 1/0 pour la classification binaire
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# 5. Tokenization (transformer les mots en entiers)
# Ici, on initialise un tokenizer pour transformer les mots en entiers.
# On limite le vocabulaire à 10 000 mots les plus fréquents, et on définit un token spécial pour les mots inconnus.
vocab_size = 10000
max_length = 200

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(df['review'])  # Apprend les mots les plus fréquents dans les reviews

# On transforme les textes en séquences d'entiers (chaque mot devient un entier selon sa fréquence)
sequences = tokenizer.texts_to_sequences(df['review'])

# 6. Padding (redre toutes les séquences de m^^eme longuerru)
# Comme les critiques n'ont pas toutes la même longueur, on complète les séquences pour qu'elles fassent toutes 200 mots
# Cela permet de traiter les données en batch dans un réseau neuronal
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

#Étape manquante dans ton dernier code : séparation des données
# On sépare les données en un ensemble d'entraînement (80%) et un ensemble de test (20%)
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['sentiment'], test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Remarque : L'embedding ne se fait pas ici directement, mais sera réalisé automatiquement 
# dans le modèle via une couche Embedding. Cette couche apprendra à transformer chaque token 
# en un vecteur dense pendant l'entraînement.
