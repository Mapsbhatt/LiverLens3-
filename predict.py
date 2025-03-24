# predict.py
import os
import numpy as np
import tensorflow as tf
import keras as keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
import sys
import zipfile
import argparse
import numpy as np
import cv2
import nibabel as nib


# 1. Dice Coefficient Metric
def dice_coefficient(y_true, y_pred):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    dice = (2. * intersection + 1.) / (tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + 1.)
    return dice

# 2. Callback for Dice Coefficient
class DiceCallback(Callback):
    def __init__(self, threshold=0.7):
        super(DiceCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        current_dice = logs.get('dice_coefficient')
        if current_dice is None:
            warnings.warn("Dice coefficient is missing in logs", RuntimeWarning)
        if current_dice > self.threshold:
            self.model.stop_training = True
            print(f"\nReached {self.threshold:.2f} Dice coefficient, stopping training.")

# 3. Learning Rate Scheduler
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=3):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    return tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1)

lr_scheduler = step_decay_schedule(initial_lr=1e-4, decay_factor=0.2, step_size=3)

# 4. Early Stopping Callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=10,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

# 5. Dice Coefficient per Channel
def dice_coefficient_per_channel(y_true, y_pred, epsilon=1e-6):
    y_pred = tf.round(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred, axis=(0,1,2))
    union = tf.reduce_sum(y_true + y_pred, axis=(0,1,2))
    dsc = (2. * intersection + epsilon) / (union + epsilon)
    return dsc

# 6. Channel Specific Dice Coefficient Metrics
def dice_coefficient_channel_0(y_true, y_pred):
    return dice_coefficient_per_channel(y_true, y_pred)[0]

def dice_coefficient_channel_1(y_true, y_pred):
    return dice_coefficient_per_channel(y_true, y_pred)[1]

def dice_coefficient_channel_2(y_true, y_pred):
    return dice_coefficient_per_channel(y_true, y_pred)[2]


MODEL_PATH = 'LiverLens3plus.h5'

# Load the model with custom objects
custom_objects = {
    'dice_coefficient': dice_coefficient,
    'dice_coefficient_channel_0': dice_coefficient_channel_0,
    'dice_coefficient_channel_1': dice_coefficient_channel_1,
    'dice_coefficient_channel_2': dice_coefficient_channel_2
}


def preprocess(volume_data):
    # Convert NIfTI to numpy array
    np_volume = volume_data.astype(np.float32)

    # Ensure the image is 3D (H, W, D)
    if len(np_volume.shape) != 3:
        raise ValueError(f"Expected a 3D volume. Received data of shape {np_volume.shape}.")

    # Normalize the volume
    np_volume /= 255.0

    # Resize each slice to (256, 256)
    processed_slices = []
    for slice_idx in range(np_volume.shape[2]):
        slice_2d = np_volume[:, :, slice_idx]
        resized_slice = cv2.resize(slice_2d, (256, 256), interpolation=cv2.INTER_AREA)
        processed_slices.append(resized_slice)

    # Convert list of slices back to a volume
    processed_volume = np.stack(processed_slices, axis=2)

    # Add batch dimension
    processed_volume = np.expand_dims(processed_volume, axis=0)

    return processed_volume


def postprocess(prediction):
    # Assuming prediction shape is (height, width, 3)
    # Take argmax over channel dimension
    grayscale_output = np.argmax(prediction, axis=-1)

    # Convert to desired labels:
    grayscale_output[grayscale_output == 1] = 128  # Liver
    grayscale_output[grayscale_output == 2] = 255  # Tumor

    # Resize to desired dimensions (if necessary)
    grayscale_output = cv2.resize(grayscale_output, (512, 512), interpolation=cv2.INTER_NEAREST)

    return grayscale_output

def predict_2d_slice(slice_2d, model):
    prediction = model.predict(slice_2d)
    return prediction

def process_zip(input_zip, output_dir, model_path):
    with zipfile.ZipFile(input_zip, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    extracted_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(('.nii', '.nii.gz'))]
    for input_image_path in extracted_files:
        output_image_path = os.path.join(output_dir, f"output_{os.path.basename(input_image_path)}")
        process_single_nifti(input_image_path, output_image_path, model_path)  # Replaced predict with process_single_nifti

def process_single_nifti(input_path, output_path, model_path):
    # Load the model
    model = tf.keras.models.load_model(model_path, custom_objects= custom_objects)
    
    # Load the NIfTI file
    input_nifti = nib.load(input_path)
    
    # Preprocess the input volume
    preprocessed_volume = preprocess(input_nifti.get_fdata())
    print(f"Shape of preprocessed_volume: {preprocessed_volume.shape}")
    
    # Check shape of the preprocessed_volume, it should be (H, W, D)
    _, _, _,depth = preprocessed_volume.shape

    # Process each slice and collect predictions
    predicted_slices = []
    for slice_idx in range(depth):
        slice_2d = preprocessed_volume[0, :, :, slice_idx]
        slice_2d = np.expand_dims(slice_2d, axis=0)  # Add batch dimension
        slice_2d = np.expand_dims(slice_2d, axis=-1)  # Add channel dimension

        # Convert the single-channel slice to 3-channel format
        slice_3_channel = np.repeat(slice_2d, 3, axis=-1)
        
        # Call your prediction function here using the 3-channel slice
        predicted_slice = predict_2d_slice(slice_3_channel, model)

        # Post-process the prediction
        predicted_slice = postprocess(predicted_slice)
        
        predicted_slices.append(predicted_slice)

    
    # Stack the predicted slices to get the full 3D volume prediction
    predicted_volume = np.stack(predicted_slices, axis=2)

    # Save the predicted volume as a new NIfTI file
    output_nifti = nib.Nifti1Image(predicted_volume, input_nifti.affine)
    nib.save(output_nifti, output_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image segmentation')
    parser.add_argument('-i', '--input', type=str, help='input file or zip containing .nii files', required=True)
    parser.add_argument('-o', '--output', type=str, help='output directory for segmented images', required=True)
    parser.add_argument('-s', '--model', type=str, help='model filename to load', required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.input.endswith('.zip'):
        process_zip(args.input, args.output, args.model)
    else:
        process_single_nifti(args.input, args.output, args.model)
