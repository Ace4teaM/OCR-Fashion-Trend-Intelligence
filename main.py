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
