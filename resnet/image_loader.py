import numpy as np
from PIL import Image

directory = '/media/anirudh/Extreme SSD1/DeepLense/augmented_dataset/sphere/translation/'
# directory = '/media/anirudh/Extreme SSD1/DeepLense/augmented_dataset/vort/rotation/9570_4.npy'
for i in range(1,5):
    array = np.load(directory+"1_%d.npy"%i)
    image = Image.fromarray(array*255)
    image.show()