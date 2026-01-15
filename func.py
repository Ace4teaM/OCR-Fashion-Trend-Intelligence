import base64
import io
from PIL import Image
import numpy as np
import requests
import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

CLASS_MAPPING = {
    "Background": 0,
    "Hat": 1,
    "Hair": 2,
    "Sunglasses": 3,
    "Upper-clothes": 4,
    "Skirt": 5,
    "Pants": 6,
    "Dress": 7,
    "Belt": 8,
    "Left-shoe": 9,
    "Right-shoe": 10,
    "Face": 11,
    "Left-leg": 12,
    "Right-leg": 13,
    "Left-arm": 14,
    "Right-arm": 15,
    "Bag": 16,
    "Scarf": 17
}

def get_image_dimensions(img_path):
    """
    Get the dimensions of an image.

    Args:
        img_path (str): Path to the image.

    Returns:
        tuple: (width, height) of the image.
    """
    original_image = Image.open(img_path)
    return original_image.size

def decode_base64_mask(base64_string, width, height):
    """
    Decode a base64-encoded mask into a NumPy array.

    Args:
        base64_string (str): Base64-encoded mask.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Single-channel mask array.
    """
    mask_data = base64.b64decode(base64_string)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_array = np.array(mask_image)
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]  # Take first channel if RGB
    mask_image = Image.fromarray(mask_array).resize((width, height), Image.NEAREST)
    return np.array(mask_image)

def create_masks(results, width, height):
    """
    Combine multiple class masks into a single segmentation mask.

    Args:
        results (list): List of dictionaries with 'label' and 'mask' keys.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Combined segmentation mask with class indices.
    """
    combined_mask = np.zeros((height, width), dtype=np.uint8)  # Initialize with Background (0)

    # Process non-Background masks first
    for result in results:
        label = result['label']
        class_id = CLASS_MAPPING.get(label, 0)
        if class_id == 0:  # Skip Background
            continue
        mask_array = decode_base64_mask(result['mask'], width, height)
        combined_mask[mask_array > 0] = class_id

    # Process Background last to ensure it doesn't overwrite other classes unnecessarily
    # (Though the model usually provides non-overlapping masks for distinct classes other than background)
    for result in results:
        if result['label'] == 'Background':
            mask_array = decode_base64_mask(result['mask'], width, height)
            # Apply background only where no other class has been assigned yet
            # This logic might need adjustment based on how the model defines 'Background'
            # For this model, it seems safer to just let non-background overwrite it first.
            # A simple application like this should be fine: if Background mask says pixel is BG, set it to 0.
            # However, a more robust way might be to only set to background if combined_mask is still 0 (initial value)
            combined_mask[mask_array > 0] = 0 # Class ID for Background is 0

    return combined_mask


def segment_image(filename):
    """
    Segmente une image en utilisant l'API Hugging Face.

    Args:
        filename (string): Chemin complet vers le fichier image (png ou jpeg)

    Returns:
        np.uint8 array: Masque de l'image
    """

    API_URL = "https://router.huggingface.co/hf-inference/models/sayeed99/segformer_b3_clothes"

    api_token = os.getenv("HUGGING_FACE_KEY")

    headers = {
        "Authorization": f"Bearer {api_token}"
        # Le "Content-Type" sera ajouté dynamiquement lors de l'envoi de l'image
    }

    with open(filename, "rb") as f:
        data = f.read()

    # détermine le type MIME de l'image
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        headers={"Content-Type": "image/jpeg", **headers}
    elif filename.endswith(".png"):
        headers={"Content-Type": "image/png", **headers}
    else:
        raise Exception("Format de fichier image inconnu : " + filename)
    
    # transmet l'image à l'API
    response = requests.post(API_URL, headers=headers, data=data)
    
    if not (response.status_code >= 200 and response.status_code < 300):
        raise Exception("Erreur de traitement de l'image : " + filename)

    # obtient les différents résultats et crée un tableau unique pour représernter le masque à plusieurs niveaux
    results = response.json()
    (width, height) = get_image_dimensions(filename)
    return create_masks(results, width, height) # np.uint8 array

def segment_images_batch(list_of_image_paths):
    """
    Segmente une liste d'images en utilisant l'API Hugging Face.

    Args:
        list_of_image_paths (list): Liste des chemins vers les images.

    Returns:
        list: Liste des masques de segmentation (tableaux NumPy).
              Contient None si une image n'a pas pu être traitée.
    """
    batch_segmentations = []

    for filename in list_of_image_paths:
        try:
            print(f"Traitement de {filename}")
            batch_segmentations.append(segment_image(filename))
            time.sleep(1)
        except Exception as e:
            print(f"Une erreur est survenue : {e}")
            batch_segmentations.append(None)

    return batch_segmentations


def display_segmented_images_batch(original_image_paths, segmentation_masks):
    """
    Affiche les images originales et leurs masques segmentés.

    Args:
        original_image_paths (list): Liste des chemins des images originales.
        segmentation_masks (list): Liste des masques segmentés (NumPy arrays).
    """

    # crée une ColorMap à partir des indices des classes
    # obtient une couleur pour chaque indice de classe à partir d'un jeu existant
    # voir couleurs disponibles : https://matplotlib.org/stable/gallery/color/colormap_reference.html
    cmap =  mpl.colormaps['tab20b'].resampled(len(CLASS_MAPPING))

    i = 0
    for image_data in segmentation_masks:
        path = original_image_paths[i]

        if image_data is None:
            continue

        i = i+1
        plt.figure(i)
        plt.suptitle(os.path.basename(path))

        # Première image
        plt.subplot(1, 2, 1)  # 1 ligne, 2 colonnes, image 1
        plt.imshow(mpimg.imread(path))
        plt.axis('off')

        # Deuxième image
        plt.subplot(1, 2, 2)  # 1 ligne, 2 colonnes, image 2
        plt.imshow(image_data, cmap=cmap)
        plt.axis('off')

        cbar = plt.colorbar(ticks=range(0,len(CLASS_MAPPING)))
        cbar.ax.set_yticklabels(CLASS_MAPPING.keys())
        
    plt.show()


def export_segmented_image(output_file, image_data):
    """
    Exporte le masque de l'image segmenté.

    Args:
        output_file (string): Chemin de l'image à créer
        image_data (NumPy array): Données de l'image
    """

    if image_data is None:
        return

    # image
    plt.imshow(image_data, cmap='gray')
    plt.axis('off')

    # pour export
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()


def export_segmented_images_batch(output_dir, original_image_paths, segmentation_masks):
    """
    Exporte les masques des images segmentés.

    Args:
        original_image_paths (list): Liste des chemins des images originales.
        segmentation_masks (list): Liste des masques segmentés (NumPy arrays).
    """

    i = 0
    for image_data in segmentation_masks:
        path = original_image_paths[i]

        if image_data is None:
            continue

        i = i+1

        # image
        plt.imshow(image_data, cmap='gray')
        plt.axis('off')

        # pour export
        plt.savefig(f"{output_dir}/{os.path.basename(path)}", bbox_inches='tight', pad_inches=0)
        plt.close()


def get_mask_bbox(image_data, mask_id):
    """
    Obtient le bounding-box du masque dans l'image

    Args:
        image_data (numpy array uint8): Données de l'image masque
        mask_id (int): id du masque (CLASS_MAPPING)

    Returns:
        tuple: (x1, y1, x2, y2) du bounding-box
    """
    ys, xs = np.where(image_data == mask_id)
    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    if xmin == ymin == xmax == ymax == 0:
        return None
    
    return (xmin, ymin, xmax + 1, ymax + 1)

def get_color_mask_bbox(image_path, rgb=[255,0,255]):
    """
    Obtient le bounding-box du masque dans l'image

    Args:
        image_data (numpy array uint8): Données de l'image masque
        mask_id (int): id du masque (CLASS_MAPPING)

    Returns:
        tuple: (x1, y1, x2, y2) du bounding-box
    """
    img = Image.open(image_path).convert("RGB")
    image_data = np.array(img)

    mask = np.all(image_data == rgb, axis=2)
    ys, xs = np.where(mask)
    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    if xmin == ymin == xmax == ymax == 0:
        return None
    
    return (xmin, ymin, xmax + 1, ymax + 1)

def cal_IoU_bbox(box1,box2):
    """
    Calcule l'IoU (Intersection over Union) entre 2 bounding-box
    IoU = Aire de l'intersection / Aire de l'union

    Avec :
        Intersection = zone commune aux deux boxes
        Union = aire totale couverte par les deux boxes

    Args:
        box1 (Tuple (x_min, y_min, x_max, y_max)): 
        box2 (Tuple (x_min, y_min, x_max, y_max)): 

    Returns:
        IoU (0-1)
    """

    x_min = 0
    y_min = 1
    x_max = 2
    y_max = 3

    # Coordonnées de l’intersection
    # le rectangle le plus petit des 2 boxs
    xA = max(box1[x_min], box2[x_min])
    yA = max(box1[y_min], box2[y_min])
    xB = min(box1[x_max], box2[x_max])
    yB = min(box1[y_max], box2[y_max])

    # Aire de l’intersection
    inter_width  = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area   = inter_width * inter_height

    # Aire de chaque bbox (w*h)
    area1 = (box1[x_max] - box1[x_min]) * (box1[y_max] - box1[y_min])
    area2 = (box2[x_max] - box2[x_min]) * (box2[y_max] - box2[y_min])

    # Aire de l’union
    union_area = area1 + area2 - inter_area

    # IoU
    return inter_area / union_area
