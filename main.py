import os
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
import base64
import io
from dotenv import load_dotenv
import func

load_dotenv()  # charge automatiquement le fichier .env

# variables globales
image_dir = "./content/images_a_segmenter"
max_images = 2

# Lister les chemins des images à traiter
# Assurez-vous d'avoir des images dans le dossier 'image_dir'!
image_paths = [f"{image_dir}/{path}" for path in os.listdir(image_dir)]

# Appeler la fonction pour segmenter les images listées dans image_paths
if image_paths:
    print(f"\nTraitement de {len(image_paths)} image(s) en batch...")
    batch_seg_results = func.segment_images_batch(image_paths[:max_images]) 
    print("Traitement en batch terminé.")
else:
    batch_seg_results = []
    print("Aucune image à traiter en batch.")


# Afficher les résultats du batch
if batch_seg_results:
    func.display_segmented_images_batch(image_paths, batch_seg_results)
else:
    print("Aucun résultat de segmentation à afficher.")