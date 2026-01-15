import os
from dotenv import load_dotenv
import func

load_dotenv()  # charge automatiquement le fichier .env

# variables globales
image_dir = "./content/images_a_segmenter"
output_dir = "./content/masks"
max_images = 1

# Lister les chemins des images à traiter
# Assurez-vous d'avoir des images dans le dossier 'image_dir'!
image_paths = [f"{image_dir}/{path}" for path in os.listdir(image_dir)]

# ignore les images déjà transformées
image_paths = [path for path in image_paths if os.path.exists(f"{output_dir}/{os.path.basename(path)}") == False]

# limite le nombre d'images
if max_images>0:
    image_paths = image_paths[:max_images]

# Appeler la fonction pour segmenter les images listées dans image_paths
if image_paths:
    print(f"\nTraitement de {len(image_paths)} image(s) en batch...")
    batch_seg_results = func.segment_images_batch(image_paths) 
    print("Traitement en batch terminé.")
else:
    batch_seg_results = []
    print("Aucune image à traiter en batch.")


# Exporte et Affiche les résultats du batch
if batch_seg_results:
    func.export_segmented_images_batch(output_dir, image_paths, batch_seg_results)
    func.display_segmented_images_batch(image_paths, batch_seg_results)
else:
    print("Aucun résultat de segmentation à afficher.")