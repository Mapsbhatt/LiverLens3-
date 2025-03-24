import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.layers import Input, UpSampling2D, MaxPooling2D, Conv2D, concatenate
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization

# Assuming you have defined the custom metrics in another file named 'custom_metrics.py'
from custom_metrics import dice_coefficient, DiceCallback, step_decay_schedule, early_stopping, dice_coefficient_per_channel, dice_coefficient_channel_0, dice_coefficient_channel_1, dice_coefficient_channel_2
from data_aug import prepare_directories, augment_data
from metrics import ALL_METRICS

metrics=['accuracy', dice_coefficient, dice_coefficient_channel_0, dice_coefficient_channel_1, dice_coefficient_channel_2]

# Extracting values (functions) from ALL_METRICS and combining with the initial metrics
combined_metrics = metrics + list(ALL_METRICS.values())

def process_and_combine_labels(image_path, preprocess_directory):
    # Load the image
    mask_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Initialize masks
    background_mask = np.zeros_like(mask_img)
    liver_mask = np.zeros_like(mask_img)
    tumor_mask = np.zeros_like(mask_img)

    # Define the regions for the masks based on pixel values
    background_mask[(mask_img >= 0) & (mask_img < 90)] = 1
    liver_mask[(mask_img >= 90) & (mask_img < 200)] = 1
    tumor_mask[mask_img >= 200] = 1

    # Merge the masks into a single 3-channel image
    combined_img = cv2.merge([background_mask, liver_mask, tumor_mask])

    # Construct the output filename
    base_filename = os.path.basename(image_path).rsplit('.', 1)[0]
    combined_path = os.path.join(preprocess_directory, base_filename + ".jpg")

    # Save the 3-channel image
    cv2.imwrite(combined_path, combined_img * 255)

# Define the function to process and combine labels
def process_and_combine(labels_dir, preprocess_directory):
    # Ensure the destination directory exists
    if not os.path.exists(preprocess_directory):
        os.makedirs(preprocess_directory)

    # Loop through all files in the directory with a progress bar from tqdm
    for filename in tqdm(os.listdir(labels_dir), desc="Processing and Combining"):
        file_path = os.path.join(labels_dir, filename)
        if file_path.endswith(".jpg"):
            process_and_combine_labels(file_path, preprocess_directory)

    print("All label images processed and combined!")

# Load and preprocess dataset function
def custom_data_generator(data_path, image_subdir, mask_subdir, batch_size=10):
    # Use Keras's built-in ImageDataGenerator for augmentation and normalization
    image_datagen = ImageDataGenerator(rescale=1./255)
    mask_datagen = ImageDataGenerator(rescale=1./255)

    # Create generators for images
    image_generator = image_datagen.flow_from_directory(
        data_path,
        classes=[image_subdir],
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode=None,
        seed=42)

    # Create generators for masks
    mask_generator = mask_datagen.flow_from_directory(
        data_path,
        classes=[mask_subdir],
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode=None,
        color_mode='rgb',
        seed=42)

    while True:
        img_batch = image_generator.next()
        mask_batch = mask_generator.next()

        red_channel = mask_batch[..., 0]
        green_channel = mask_batch[..., 1]
        blue_channel = mask_batch[..., 2]

        # Convert to binary format
        red_binary = (red_channel > 0.5).astype(np.uint8)
        green_binary = (green_channel > 0.5).astype(np.uint8)
        blue_binary = (blue_channel > 0.5).astype(np.uint8)

        # Stack them together for one-hot encoded format
        mask_batch_binary = np.stack([blue_binary, green_binary, red_binary], axis=-1)

        yield img_batch, mask_batch_binary


def load_data(train_path="/content/to/train/", 
              val_path="/content/to/val/",
              batch_size=10):
    train_gen = custom_data_generator(train_path, "images", "labels", batch_size) #Ensure to provide the preprocessed labels to the input. 
    val_gen = custom_data_generator(val_path, "images", "labels", batch_size)   #Ensure to provide the preprocessed labels to the input.

    train_images, train_labels = next(train_gen)
    val_images, val_labels = next(val_gen)

    return (train_images, train_labels), (val_images, val_labels)


src_image_folder = '/content/to/image'
src_label_folder = '/content/to/label'

train_image_folder = '/content/to/train/images'
val_image_folder = '/content/to/val/images'
test_image_folder = '/content/to/test/images'

train_label_folder = '/content/to/train/labels'
val_label_folder = '/content/to/val/labels'
test_label_folder = '/content/to/test/labels'

#Prepare the directories
prepare_directories(train_image_folder, val_image_folder, test_image_folder, train_label_folder, val_label_folder, test_label_folder, src_image_folder, src_label_folder)
#Augment the Data for traina and val set.
augment_data(3, train_image_folder, train_label_folder)
augment_data(3, val_image_folder, val_label_folder)

# Specify the directories for label images and the destination directory
preprocess_train_directory = "/content/to/preprocessed_train_label"
preprocess_val_directory = "/content/to/preprocessed_val_label"

# Call the function to process and combine labels
process_and_combine_labels(train_label_folder, preprocess_train_directory)
process_and_combine_labels(val_label_folder, preprocess_val_directory)



#Multi-Attention Mechanism
def multi_attention_block(input_feature, path_feature):
    g = tf.keras.layers.Conv2D(filters=input_feature.shape[-1], kernel_size=1)(input_feature)

    # Upsample g to have the same spatial dimensions as path_feature
    g = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(g)

    x = tf.keras.layers.Conv2D(filters=input_feature.shape[-1], kernel_size=1)(path_feature)
    psi = tf.keras.activations.relu(g + x, alpha=0.0)
    psi = tf.keras.layers.Conv2D(1, kernel_size=1)(psi)
    psi = tf.keras.activations.sigmoid(psi)
    return path_feature * psi

#Deptwise Convolution Block
def depthwise_conv_block(tensor, n_filters):
    tensor = tf.keras.layers.SeparableConv2D(filters=n_filters, kernel_size=(3, 3), padding='same')(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = tf.keras.activations.relu(tensor)

    tensor = tf.keras.layers.SeparableConv2D(filters=n_filters, kernel_size=(3, 3), padding='same')(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = tf.keras.activations.relu(tensor)
    return tensor

#Unet3+ 
def unetpp(input_shape=(256, 256, 3), num_classes=3):
    inputs = Input(shape=input_shape)

    # Contracting/downsampling path
    c1 = depthwise_conv_block(inputs, 64)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = depthwise_conv_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = depthwise_conv_block(p2, 256)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = depthwise_conv_block(p3, 512)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = depthwise_conv_block(p4, 1024)
    c5 = multi_attention_block(c5, c4)  # Added attention here

    # Nested skip pathways
    z1_1 = concatenate([c5, c4])
    z1_1 = depthwise_conv_block(z1_1, 512)
    z1_1 = multi_attention_block(z1_1, c3)  # Added attention here

    #u7 = UpSampling2D((2, 2))(z1_1)
    z2_1 = concatenate([z1_1, c3])
    z2_1 = depthwise_conv_block(z2_1, 256)
    z2_1 = multi_attention_block(z2_1, c2)  # Added attention here

    #u8 = UpSampling2D((2, 2))(z2_1)
    z3_1 = concatenate([z2_1, c2])
    z3_1 = depthwise_conv_block(z3_1, 128)

    # More nested skip pathways
    upsamp_c3 = UpSampling2D((2, 2))(c3)
    z2_2 = concatenate([z2_1, upsamp_c3, z3_1])
    z2_2 = depthwise_conv_block(z2_2, 256)

    u10 = UpSampling2D((2, 2))(z2_2)
    upsamp_c2 = UpSampling2D((2, 2))(c2)
    upsamp_z3_1 = UpSampling2D((2, 2))(z3_1)
    z3_2 = concatenate([u10, upsamp_c2, upsamp_z3_1])
    z3_2 = depthwise_conv_block(z3_2, 128)

    # Yet more nested skip pathways
    upsamp_c4 = UpSampling2D((2, 2))(c4)
    mp_z2_2 = MaxPooling2D((2, 2))(z2_2)
    z1_2 = concatenate([z1_1, upsamp_c4, mp_z2_2])
    z1_2 = depthwise_conv_block(z1_2, 512)


    mp_z3_2 = MaxPooling2D((4, 4))(z3_2)
    mp_z2_2_1 = MaxPooling2D((2, 2))(z2_2)
    z2_3 = concatenate([z1_2, c3, mp_z2_2_1, mp_z3_2])
    z2_3 = depthwise_conv_block(z2_3, 256)

    u13 = UpSampling2D((2, 2))(z2_3)
    mp_z3_2 = MaxPooling2D((2, 2))(z3_2)
    z3_3 = concatenate([u13, c2, mp_z3_2, z3_1])
    z3_3 = depthwise_conv_block(z3_3, 128)
    z3_3 = UpSampling2D((2, 2))(z3_3)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(z3_3)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    return model


# Train the model function
def train_model():
    (train_images, train_labels), (val_images, val_labels) = load_data()
    
    model = unetpp()

    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',  
                  metrics = combined_metrics)


    checkpoint_filepath = '/content/to/dir/Final_unetpp.h5'
    model_checkpoint_callback = ModelCheckpoint(
                            filepath=checkpoint_filepath,
                            save_weights_only=True,
                            monitor='val_loss',
                            mode='min',
                            save_best_only=True,
                            verbose=1
)
    
    # Training the model
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs= 100,  
        batch_size= 10,  
        callbacks=[model_checkpoint_callback]
    )

    # Save the model after training
    model.save('/app/saved_model/my_model')
    model.save_weights('/app/saved_model/my_model_weights')

if __name__ == '__main__':
    train_model()
