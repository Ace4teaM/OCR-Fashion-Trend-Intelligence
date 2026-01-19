# Fashion trend intelligence

Ce projet est composé de 3 programmes:
* main.py : Réalise le traitement des images du dossier `content/images_a_segmenter` et exporte les masques dans `content/masks` (seul les images non traité sont prises en compte)
* perfs.py : Réalise un test de performances sur le jeu d'essai présent dans le dossier `perfs/images` (le resultat est écrit dans un fichier texte avec timestamp)
* validation.py : Réalise un test de validité du système en comparant le résultat créé avec le résultat attendu. le résultat est une note de type IoU

