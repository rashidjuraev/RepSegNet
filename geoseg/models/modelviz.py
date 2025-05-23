import sys
import os
import torch
from torchviz import make_dot
from MixerUnet import ConvMixer_UNET_down


# Create a dummy input tensor with the same dimensions as your model's input
dummy_input = torch.randn(1, 3, 256, 256)

# Instantiate your model
model = ConvMixer_UNET_down(2)

# Pass the dummy input through the model
output = model(dummy_input)

# Use make_dot to create a visualization of the model
dot = make_dot(output, params=dict(model.named_parameters()))

# Save or render the visualization
dot.render("model_visualization", format="png")
dot.view()