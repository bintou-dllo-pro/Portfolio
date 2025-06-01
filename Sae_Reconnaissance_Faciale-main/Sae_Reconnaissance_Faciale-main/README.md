# Sae_Reconnaissance_Faciale


# HAJBI Khaoula
# DIALLO Bintou


# Introduction : 

Notre professeur, Monsieur CABESSA, nous a fourni un code permettant d’effectuer une reconnaissance faciale en temps réel. Nous avons dû débugger et adapter ce code afin qu’il fonctionne pour nous puis, une fois ce code fonctionnel de notre côté, nous avons cherché plusieurs façons de le réadapter de plusieurs façons différentes selon la demande de notre professeur. Nous désignerons chacune des demandes de notre professeur comme étant des étapes, une étape correspondant à l’une des consignes données.


# Technologies utilisées : 

Nous avons choisi de travailler sur Google Colab car il s’agissait de l’environnement de travail sur lequel nous étions les plus à l’aise pour coder en python.

Voici toutes les librairies que nous avons utilisées pour nos différents codes ainsi qu’une description de leurs fonctions : 

- Pickle : Permet de sérialiser et désérialiser des structures de données en Python

- Numpy : Pour le calcul scientifique en Python qui permet de manipuler des tableaux multidimensionnels

- Os : Permet d'interagir avec le système d'exploitation (manipulation de chemins de fichiers, etc)

- CV2 :  OpenCV (Open Source Computer Vision Library) pour le traitement d'images par ordinateur. Permet la manipulation d'images et la détection de visages.

- De sklearn.tree, DecisionTreeClassifier : DecisionTreeClassifier pour la classification de données avec arbre de décision.

- De sklearn.model_selection, train_test_split: Pour diviser un ensemble de données en sous-ensembles d'entraînement et de test.

- De IPython.display, display, Javascript : Pour exécuter du JavaScript dans le notebook Google Colab.

- De google.colab.output, eval_js: Pour exécuter du JavaScript dans une cellule Google Colab et récupérer le résultat en Python.

- De base64, b64decode: Pour décoder des données encodées en base64.

- De sklearn.linear_model, LogisticRegression : pour effectuer des régressions logistiques

- Matplotlib.pyplot : Permet de créer des graphiques et des visualisations de données

- De sklearn.neighbors, KNeighborsClassifier : Pour utiliser l’algorithme KNN

- De tensorflow.keras.models, Sequential : Pour créer des modèles de réseaux de neurones en empilant des couches

- De tensorflow.keras.models, load_model :  pour charger un modèle Keras 

- De tensorflow.keras.layers, Conv2D, MaxPooling2D, Flatten, Dense, Dropout : 
Pour importer plusieurs types de couches utilisés dans les réseaux de neurones convolutifs 

- De tensorflow.keras.optimizers, Adam : Méthode de descente de gradient stochastique qui ajuste le taux d'apprentissage

- De sklearn.preprocessing, LabelBinarizer : Classe de prétraitement pour transformer des étiquettes de classe en vecteurs binaires


# Etape 1 : 

Dans un premier temps, nous avons fait en sorte de comprendre le code fourni par notre professeur et débuggé ce même code tout comme nous l’avons mentionné précédemment. Bien qu’il fonctionnait pour notre professeur, il ne fonctionnait pas pour nous car il n' activait pas la prise d’image à l’aide de notre webcam qui était primordiale pour que nous puissions exécuter la reconnaissance faciale. Nous avons, pour remédier à ce problème, réalisé un script en Javascript qui nous permettait d’utiliser la webcam de notre ordinateur et l’avons intégré aux deux notebooks fournis par notre professeur. 

Une fois cette modification faite nous avons adapté ce code pour effectuer les étapes suivantes.

# Etape 2 : 
Nous avons choisi d’adapter notre deuxième notebook aux algorithmes suivants : L’algorithme de régression Logistique et l’algorithme d’arbre de décision dans un cas de classification.


Pour le cas de la régression linéaire, on utilise une fonction JavaScript pour capturer des images via une webcam dans Google Colab. Les visages sont détectés dans ces images à l'aide d'OpenCV et d'un modèle de cascade de Haar, en redimensionnant les images pour uniformiser la taille des entrées. Les images traitées sont converties en vecteurs et associées à des étiquettes, divisées en ensembles d'entraînement et de test, avec l'ajout d'une fausse classe. Un modèle de régression logistique est ensuite entraîné pour classer les visages, et ce modèle est utilisé pour identifier les visages dans les images que nous capturons dans un second temps. Nous parvenons ensuite à effectuer la reconnaissance en temps réel.

Pour le cas d’arbre de décision dans un cas de classification : Les étapes de débutsont similaires à celles effectuées pour notre régression linéaire. Suite à ces dernières, un arbre de décision est entraîné sur ces données pour classifier les visages. Puis, le code capture une nouvelle image, détecte et traite un visage de la même manière que précédemment, et utilise l'arbre de décision entraîné pour prédire l'identité du visage capturé.


# Etape 3 : 

Nous avions pour objectif d’effectuer une reconnaissance faciale mais il fallait que notre code puisse indiquer si une personne est “admise” ou “non-admise” en se basant sur le fait que nous avons enregistré une personne comme étant celle qui est admise et une autre comme étant celle qui n’est pas admise.

Le début de notre code est similaire à celui des codes mentionnés précédemment.
On répète le procédé de capture pour collecter des images pour la personne admise et la personne non admise puis on stocke les visages et leurs étiquettes localement.On utilise le modèle KNeighborsClassifier de scikit-learn, qui attribue une catégorie ou une classe spécifique pour identifier si un visage appartient à la catégorie "admis" ou "non admis". Il classifie en analysant et en comparant le visage avec ceux qu'il a déjà appris durant la phase d'entraînement, en se basant sur la similarité avec les k visages les plus proche. 



# Etape 4 : 

Il nous a été demandé, dans cette partie, de mettre en place un algorithme de classification par réseau de neurones : 

Pour ce faire, nous avons procédé ainsi : 
Le début de notre code est similaire à celui des codes mentionnés précédemment.
Les données sont encodées en vecteurs binaires puis divisées en ensembles d'entraînement et de test. Le modèle de réseau de neurones dense (chaque neurone d'une couche est connecté à tous les neurones de la couche précédente et de la couche suivante) est configuré avec l'optimiseur Adam, et est soumis à un entraînement en utilisant les données. Après l'entraînement, le modèle et le transformateur de données sont archivés pour une futur utilisation. 


# Etape 5 :
Il nous a été demandé, dans cette partie, de mettre en place un réseau de neurones convolutionnels (CNN) : 

En apprentissage automatique, un réseau de neurones convolutionnel est un type de réseau de neurones artificiels dans lequel le motif de connexion entre les neurones est inspiré par le cortex visuel des animaux : certaines cellules sont spécialisées pour répondre à des zones spécifiques du champ visuel. Le réseau de neurones convolutionnel est alors plus spécialisé dans la détection d’image qu’un réseau de neurones simple.

Pour mettre en place ce modèle, le début du code reste similaire à celui des précédents. Ces données sont normalisées et préparées pour l'entraînement d'un modèle CNN, conçu pour reconnaître différents visages.Le modèle CNN utilise des couches pour traiter les images de visages : les couches convolutionnelles détectent des caractéristiques visuelles, les couches de pooling diminuent le nombre de variables ou caractéristiques traitées par le modèle tout en préservant les caractéristiques importantes, la couche de flattening transforme les données en un vecteur unidimensionnel, et les couches denses réalisent la classification en se basant sur les caractéristiques extraites. Après avoir préparé les images et les noms, le modèle est entraîné pour reconnaître les visages.


# Conclusion :

Ce projet fut un projet complet qui nous a permis de découvrir et manipuler de nombreux modèles de Machine Learning et nous a exercé à réadapter un code déjà existant.
