import os
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
import base64
import io
from dotenv import load_dotenv

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
    with open(filename, "rb") as f:
        data = f.read()
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        this_headers={"Content-Type": "image/jpeg", **headers}
    elif filename.endswith(".png"):
        this_headers={"Content-Type": "image/png", **headers}
    else:
        raise Exception("Format de fichier image inconnu : " + filename)
    
    response = requests.post(API_URL, headers=this_headers, data=data)
    return response.json()

output = query(f"{image_dir}/{image_paths[0]}")
print("response",output)