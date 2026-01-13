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
max_images = 3  # Commençons avec peu d'images

# charge le token Huging face
api_token = os.getenv("HUGGING_FACE_KEY")

API_URL = "https://router.huggingface.co/hf-inference/models/sayeed99/segformer_b3_clothes"
headers = {
    "Authorization": f"Bearer {api_token}"
    # Le "Content-Type" sera ajouté dynamiquement lors de l'envoi de l'image
}

# Lister les chemins des images à traiter
# Assurez-vous d'avoir des images dans le dossier 'image_dir'!
image_paths = os.listdir(image_dir)

if not image_paths:
    print(f"Aucune image trouvée dans '{image_dir}'. Veuillez y ajouter des images.")
else:
    print(f"{len(image_paths)} image(s) à traiter : {image_paths}")


def query(filename):
    try:
        with open(filename, "rb") as f:
            data = f.read()
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            this_headers={"Content-Type": "image/jpeg", **headers}
        elif filename.endswith(".png"):
            this_headers={"Content-Type": "image/png", **headers}
        else:
            raise Exception("Format de fichier image inconnu : " + filename)
        
        response = requests.post(API_URL, headers=this_headers, data=data)

        if not (response.status_code >= 200 and response.status_code < 300):
            raise Exception("Erreur de traitement de l'image : " + filename)

        results = response.json()
        (width, height) = func.get_image_dimensions(filename)
        image_data = func.create_masks(results, width, height)
        
        for result in results:
            print('score', result['score'])
            print('label', result['label'])

    except Exception as e:
        print(f"Une erreur est survenue : {e}")

for filename in image_paths:
    print(f"{image_dir}/{filename}")
    query(f"{image_dir}/{filename}")
    break# un seul pour le moment

print("finish")
