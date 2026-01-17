import pandas as pd
import numpy as np
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


#1. Chargement des données
#on charge le fichier CSV contenant les critiques de films et leurs sentiments (positif ou négatif)
df = pd.read_csv("IMDB Dataset.csv")
print(df.head())

#2. verif basique des données
# verification des vaaleurs manquantes et observation de la repartition des classes
print(df.info())
print(df['sentiment'].value_counts())

#3. Nettoyage du texte
"""Creation d'une fonction de nettoryage des reviews:
- suppression des balises HTML
- suppression des caractères speciaux
- conversion en minuscules"""

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # supprimer HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # garder que lettres
    text = text.lower()
    return text

df['review'] = df['review'].apply(clean_text)

#4.Encodage des labels
#conversion des labels "positive"/"negative" en 1/0 pour la classification binaire
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

#5. Tokenization (transformer les mots en entiers)
""" initialissation d'un tokenizer pour transformer les mots en entiers.
On va limiter le vocabulaire aux 10 000 mots les plus frequentss, et on va defninir un token spécial pour ls mots inconnus"""
vocab_size = 10000
max_length = 200

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(df['review'])  #permet de definir quels sont les mots les plus fréquents dans les reviews

#Les textes sont transformés en séquences d'entiers ( de par le fait que chaque mot devient un entier choisi selon sa fréquence)
sequences = tokenizer.texts_to_sequences(df['review'])

#6. Padding (rendre toutes les séquences de m^^eme longuerru)
"""Etant donné que toutes les critiques n'ont pas forcément la même longueur on va modifier la longueru pour qu'elles faassent toutes 200 mots
Cela permet de traiter les données en batch dans un réseau neuronal"""
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

#Etape de séparation des données
#on sépare les donnes en deux ensembles : l'un qui srvira à l'entraînement et l'auter à la réalisation de tests :  (entraînement : 80%, contre 20% pour les tests)
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['sentiment'], test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Paramètres du modèle (permettent d'adapter le modèle aux types de données et d'adapter sa complexité, influence sur laperformance et le temps d'entraînemnt)
vocab_size = 10000 #nombre maximal de mots distincts pris en compte dans le vocabulaire (ici on prend les 10 000 mots les plus fréquents), permettant aainsi d'accorder moins d'importance aux mots trop rares
max_length = 200#longueur maximale de chaque séquence de texte pour une uniformalité des longueurs 

#7. Création du modèle séquentiel
# Le modèle commence par une couche Embedding qui trasnforme chaqe mot en vecteur dense
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length), #ici on définiti le nombre de mots différents dans mon vocabulaire (10 000 plus fréquents), Chaque mot est encodé en un vecteur dense de 128 dimensions, que Chaque séquence de texte fait 200 tokens (mots max)
    tf.keras.layers.GRU(64, return_sequences=False), #MODIF : GRU avec 64 unités (=neurones), plus léger que LSTM mais aussi adapté aux données séquentielles
    tf.keras.layers.Dropout(0.5),#dropout pour éviter ce qu'on appelle l’overfitting : on cache une parie des données aléaatoirement comme ça on évite le surapprentissage car le modèle apprned à ne pas dépendre de certaines connexions
    tf.keras.layers.Dense(1, activation='sigmoid')#sortie binaire : 0 ou 1 (négatif/positif), on utilise la fonction d'activation sigmoïde
])

#8. Compilation du modèle
# On utilise l'optimiseur 'adam' et la fonction de perte 'binary_crossentropy' pour une classification binaire
model.compile(loss='binary_crossentropy',
              optimizer='adam',#optimiseur qui s'adapte en ajustant les poids por minimiser la perte
              metrics=['accuracy'])

#9. Affichage du résumé du modèle, visualisation des couches du réseau, de leur type, de leur taille, et de leur nombre total de paramètres
model.summary()


# Entraînement du modèle

"""On entraîne le modèle sur X_train et y_train, avec 20% de validation (càd que 20% des données d'netrainement sont utiliséss pour valider la performance à whaque epoch)
epochs = nb de fois où on aprcours le dataset
batch_size = nombre d'exemples vus à la fois"""
history = model.fit(
    X_train, y_train,
    epochs=10,               # MODIF  : passage de 5 à 10
    batch_size=100,          # MODIF  : passage de 64 à 100
    validation_split=0.5   # MODIF  : 0.2 à 0.5
)

# Affichage des courbes d'apprentissage

# Affichage de la courbe d'accuracy
plt.plot(history.history['accuracy'], label='Accuracy (train)')
plt.plot(history.history['val_accuracy'], label='Accuracy (validation)')
plt.title('Évolution de l’accuracy')
plt.xlabel('Époque')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Affichage de la courbe de perte (loss)
plt.plot(history.history['loss'], label='Loss (train)')
plt.plot(history.history['val_loss'], label='Loss (validation)')
plt.title('Évolution de la loss')
plt.xlabel('Époque')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# Évaluation finale sur les données de test

#Le modèle est testé sur le set de test jamais vu pendant le training
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")



#-----------------------------------------------------------------------------------Prédictions

#prédiction des proba (valeurs entre 0 et 1)
y_pred_probs = model.predict(X_test)

#des proba selons les classes : 0.5 ou + -> positif (1), sinon c'est négatif (0)
y_pred = (y_pred_probs > 0.5).astype(int)

# Rapport de classification : précision, rappel, F1-score
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))
"""
12. Rapport de classification
On affiche des métriques permettat d'évaluer les performances de notre modèole :
- la pécision : nb de prédictions qui étaienty reellement correctes parmi les trouvées positives
- Rappel (recall) : parmi les vrais positifs, combien ont été bien prédits ?
- F1-score : moyenne entre précision et rappel
s'ajoutent à l'accuracy pour l'éval du modele
"""

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)

# Visualisation de la matrice avec seaborn
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Négatif', 'Positif'], yticklabels=['Négatif', 'Positif'])
plt.xlabel('Prédictions')
plt.ylabel('Vraies classes')
plt.title('Matrice de confusion')
plt.show()

"""# 13. Matrice de confusion
Une matrice de confusion permet de visualiser les erreus faites par oe modèle lors fe la classifacation : 
- nombre de vrais positifs, faux positifs, vrais négatifs, faux négatifs.
permt de voir si le modèle confond souvent les classes (inversion positif et negatif)
"""