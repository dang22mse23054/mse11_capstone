
import sys,os, random
sys.path.append('../')

import Augmentor
import matplotlib.pyplot as plt

FODLER_PATH = 'raw/fer2013-org/train'

from common.constants import Constants
EMOTION = Constants.Emotion()

def augment_data(folder, num_of_images, ):
    if (int(num_of_images) > 0):
        p = Augmentor.Pipeline(folder, './')

        # p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        p.flip_left_right(probability=0.5)
        p.random_contrast(probability=0.5, min_factor=0.5, max_factor=1.5)
        p.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.3)

        p.sample(num_of_images)

        print(f"{num_of_images} augmented images have been saved to {folder}.")
        
subfolder_list = os.listdir(FODLER_PATH)
for folder_name in subfolder_list:
    print(folder_name)
    num_of_augumented_imgs = EMOTION.Augmentor.get(folder_name, 0)
    folder_path = f'{FODLER_PATH}/{folder_name}'
    print(f'{folder_name} - {num_of_augumented_imgs} - {folder_path}')
    augment_data(folder_path, num_of_augumented_imgs)
    
# plot_num_of_images()