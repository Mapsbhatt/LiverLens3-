from cProfile import label
import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# Define your source folder and destination folders
def prepare_directories(train_image_folder, val_image_folder, test_image_folder, train_label_folder, val_label_folder, test_label_folder, src_image_folder, src_label_folder):

    if not os.path.exists(train_image_folder):
        os.makedirs(train_image_folder)

    if not os.path.exists(val_image_folder):
        os.makedirs(val_image_folder)

    if not os.path.exists(test_image_folder):
        os.makedirs(test_image_folder)

    if not os.path.exists(train_label_folder):
        os.makedirs(train_label_folder)

    if not os.path.exists(val_label_folder):
        os.makedirs(val_label_folder)

    if not os.path.exists(test_label_folder):
        os.makedirs(test_label_folder)

    # Split into training, validation, and testing sets (70% training, 15% validation, 15% testing)
    all_images = os.listdir(src_image_folder)
    train_images, test_images = train_test_split(all_images, test_size=0.30, random_state=42)
    val_images, test_images = train_test_split(test_images, test_size=0.50, random_state=42)

    # Move the split files into respective directories
    for image in train_images:
        shutil.move(os.path.join(src_image_folder, image), train_image_folder)
        shutil.move(os.path.join(src_label_folder, image), train_label_folder)

    for image in val_images:
        shutil.move(os.path.join(src_image_folder, image), val_image_folder)
        shutil.move(os.path.join(src_label_folder, image), val_label_folder)

    for image in test_images:
        shutil.move(os.path.join(src_image_folder, image), test_image_folder)
        shutil.move(os.path.join(src_label_folder, image), test_label_folder)


#Data Augmentor

def augment_data(b_size, img_folder, label_folder ifi=6):
    data_gen_args = dict(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
    print('Creating augmented training images...')
    seed = 1337
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    save_here_img = img_folder  # Save augmented images in the original image folder
    save_here_mask = label_folder  # Save augmented labels in the original label folder

    train_image_files = sorted(os.listdir(img_folder))
    train_mask_files = sorted(os.listdir(label_folder))

    k = 0
    for i in tqdm(range(len(train_image_files)), desc="Augmenting"):
        normalimgPath = os.path.join(img_folder, train_image_files[i])
        normalmaskPath = os.path.join(label_folder, train_mask_files[i])

        img = np.expand_dims(plt.imread(normalimgPath), 0)
        mask = np.expand_dims(plt.imread(normalmaskPath), 0)

        for x, y, val in zip(
            image_datagen.flow(img, batch_size=b_size, seed=seed, save_to_dir=save_here_img, save_prefix='aug_{}'.format(str(k)), save_format='jpg'),
            mask_datagen.flow(mask, batch_size=b_size, seed=seed, save_to_dir=save_here_mask, save_prefix='aug_{}'.format(str(k)), save_format='jpg'), range(ifi)
        ):
            k += 1




