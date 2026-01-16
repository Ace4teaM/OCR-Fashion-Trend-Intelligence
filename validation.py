import os
from dotenv import load_dotenv
import func
import re
from PIL import Image
import numpy as np

load_dotenv()  # charge automatiquement le fichier .env

# variables globales
image_dir = "./validation/test"
output_file = f"{image_dir}/result.png"

# Lister les chemins des images à traiter
# Assurez-vous d'avoir des images dans le dossier 'image_dir'!
image_paths = [f"{image_dir}/{path}" for path in os.listdir(image_dir) if re.search(r"^image.(?:png|jpg|jpeg)$", os.path.basename(path))]
mask_paths = [f"{image_dir}/{path}" for path in os.listdir(image_dir) if re.search(r"^mask_[\d+].(?:png|jpg|jpeg)$", os.path.basename(path))]

# vérification des images en entrée
if len(image_paths) != 1:
    print("Aucune image à traiter ou plusieurs images présentes.")
    exit(1)

if len(mask_paths) == 0:
    print("Aucune image de masque pour traitement du résultat.")
    exit(1)

# Traitement
print(f"\nTraitement de l'image {image_paths[0]}")
# Appeler la fonction pour segmenter l'image en entrée
seg_results = func.segment_image(image_paths[0])
# Exporte l'image pour visualisation
func.export_segmented_image(output_file, seg_results)


# Test chaque masque de l'image
IoUs = []
for path in mask_paths:
    # obtient l'id du masque recherché
    filename = os.path.basename(path)
    mask_id = int(filename[5:filename.rfind('.')]) # "mask_*.ext"
    print("filename", filename)
    print("mask_id", mask_id)
    # obtient le bounding-box du masque resultat
    result_bbox = func.get_mask_bbox(seg_results, mask_id)
    print("result_bbox",result_bbox)
    # si le masque n'existe pas on considère que c'est un raté
    if result_bbox is None:
        IoUs.append(0.0)
        print(f"mask {mask_id} introuvable")
        continue
    # obtient le bounding-box du masque attendu
    expected_bbox = func.get_color_mask_bbox(path)
    print("expected_bbox",expected_bbox)
    # calcule de l'IoU
    IoUs.append(func.cal_IoU_bbox(result_bbox, expected_bbox))

print("IoUs",IoUs)
print("Resultat",round(sum(IoUs) / len(IoUs), 2))