import os
from dotenv import load_dotenv
import func
import time

load_dotenv()  # charge automatiquement le fichier .env

# variables globales
image_dir = "./perfs/images"
max_images = 50

# Lister les chemins des images à traiter
# Assurez-vous d'avoir des images dans le dossier 'image_dir'!
image_paths = [f"{image_dir}/{path}" for path in os.listdir(image_dir)]

# limite le nombre d'images
if max_images>0:
    image_paths = image_paths[:max_images]

# Segmente les images une à une et enregistre les performances
durations = [] # temps de traitement pour chaque image
scores = [] # score moyen pour chaque image
output = []
for filename in image_paths:
    print(f"\nTraitement de {filename}...")
    infos={}
    func.segment_image(filename, infos)
    time.sleep(1)
    durations.append(infos['duration'])
    scores.append(infos['score'])
    print(filename, infos)
    output.append(f"{os.path.basename(filename)}, {infos}\n")


print(f"performances pour {len(durations)} images")
print("scores", scores)
print("durations", durations)
print(f"durée moyenne", round(sum(durations) / len(durations), 2), "sec")
print(f"score moyen", sum(scores) / len(scores))

with open(f"./perfs/{len(durations)}-samples-{time.time()}.txt", "w", encoding="utf-8") as f:
    f.write(f"performances pour {len(durations)} images\n")
    f.write(f"> durée moyenne {round(sum(durations) / len(durations), 2)} sec \n")
    f.write(f"> score moyen {sum(scores) / len(scores)} \n")
    f.write("\n\n")
    for line in output:
        f.write(line)