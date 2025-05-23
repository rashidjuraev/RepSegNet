#%%
import numpy as np
import matplotlib.pyplot as plt

# Load the saved numpy arrays
inputs = np.load('/data1/home/ict08/skinseg/GeoSeg/geoseg/models/results/isic_unet_inputs.npy')
predictions = np.load('/data1/home/ict08/skinseg/GeoSeg/geoseg/models/results/isic_unet_predictions.npy')
ground_truths = np.load('/data1/home/ict08/skinseg/GeoSeg/geoseg/models/results/isic_unet_ground_truth.npy')

# Print shapes to verify
print(f"Shape of inputs: {inputs.shape}")
print(f"Shape of predictions: {predictions.shape}")
print(f"Shape of ground truths: {ground_truths.shape}")
#%%
# Function to display an image, its prediction, and ground truth
def show_results(index):
    input_image = inputs[index].transpose(1, 2, 0)  # Convert from CHW to HWC format
    prediction = np.argmax(predictions[index], axis=0)  # Take argmax to get predicted class (H, W)
    ground_truth = ground_truths[index]  # Ground truth is already in (H, W) format

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(input_image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Prediction")
    plt.imshow(prediction, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Ground Truth")
    plt.imshow(ground_truth, cmap='gray')
    plt.axis('off')

    plt.show()

# Show a few sample results
for i in range(31):  # Display the first 5 results
    show_results(i)
#%%