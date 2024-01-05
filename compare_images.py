import cv2
from skimage.metrics import structural_similarity as ssim

def calculate_image_similarity(image1_path, image2_path):
    # Charger les images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Redimensionner les images pour qu'elles aient la même taille
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Convertir les images en niveaux de gris (pour le SSI)
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculer l'indice structural de similarité (SSI)
    similarity_index, _ = ssim(gray_image1, gray_image2, full=True)

    return similarity_index



image1_path = './first.png'
image2_path = './second.png'

similarity = calculate_image_similarity(image1_path, image2_path)
print(f"Indice de similarité : {similarity}")
