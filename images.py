import jax
import jax.numpy as jnp

import numpy as np
from PIL import Image

import streamlit as st
from core import NN
from core import (
    siren_first_layer_init,
    siren_other_layers_init,
    he_normal_init,
    he_uniform_init,
    lecun_normal_init,
    lecun_uniform_init,
    zeros_init,
)
from core import fit, predict

import flax.linen as nn

initializers = {key: value for key, value in locals().items() if key.endswith("_init")}

st.title("SIREN for Images")
st.subheader("Reconstruction")

# create sliders for the hyperparameters
img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# access the uploaded image as a numpy array
if img is not None:
    img = Image.open(img)
    img = np.array(img) / 255.0

    # display the uploaded image
    st.image(img, caption=f"Image shape: {img.shape}")

    # convert to grayscale
    grayscale = st.checkbox("Convert to grayscale")
    if grayscale:
        img = np.mean(img, axis=-1, keepdims=True)
        st.image(img, caption=f"Image shape: {img.shape}")

    # scale down the image
    scale_down = st.number_input("Scale down image by this factor", value=1)
    if scale_down > 1:
        img = nn.avg_pool(
            img, (scale_down, scale_down), strides=(scale_down, scale_down)
        )
        # show image in the center of the page (instead of to the left)

        st.image(
            np.array(img),
            caption=f"Image shape: {img.shape}",
            use_column_width=True,
        )

    # create 3 columns for the hyperparameters
    col1, col2, col3, col4 = st.columns(4)
    n_hidden_layer_neurons = col1.number_input("Neurons", value=256)

    n_layers = col2.number_input("Layers", value=4)
    n_hidden_layer_neurons = [n_hidden_layer_neurons] * n_layers

    activation_name = col3.selectbox(
        "Activation",
        ["relu", "sigmoid", "tanh", "softplus", "gelu", "sine"],
    )
    factor = col4.number_input(
        "Activation scaling factor", value=30 if activation_name == "sine" else 1
    )
    activation = (
        lambda x: getattr(nn.activation, activation_name)(factor * x)
        if activation_name != "sine"
        else jnp.sin(factor * x)
    )

    # define the initializers
    col1, col2 = st.columns(2)
    kernel_first_layer_init = col1.selectbox(
        "Kernel first layer init",
        list(initializers.keys()),
        0 if activation_name == "sine" else 4,
    )
    kernel_first_layer_init = initializers[kernel_first_layer_init]

    bias_first_layer_init = col2.selectbox(
        "Bias first layer init",
        list(initializers.keys()),
        0 if activation_name == "sine" else 6,
    )
    bias_first_layer_init = initializers[bias_first_layer_init]
    col1, col2 = st.columns(2)
    kernel_other_layers_init = col1.selectbox(
        "Kernel other layers init",
        list(initializers.keys()),
        1 if activation_name == "sine" else 4,
    )
    kernel_other_layers_init = initializers[kernel_other_layers_init]

    bias_other_layers_init = col2.selectbox(
        "Bias other layers init",
        list(initializers.keys()),
        1 if activation_name == "sine" else 6,
    )
    bias_other_layers_init = initializers[bias_other_layers_init]

    # define the optimizer
    col1, col2, col3, col4 = st.columns([1, 1.5, 1, 1])
    lr = col1.number_input("Learning rate", value=1e-4, format="%.8f")
    batch_size = col2.number_input(
        f"Batch size (full size = {img.shape[0]*img.shape[1]})",
        img.shape[0] * img.shape[1],
    )
    key = col3.number_input("Seed", value=0)
    key = jax.random.PRNGKey(key)
    n_iters = col4.number_input("Iterations", value=100)

    col1, col2 = st.columns(2)
    x_min = col1.number_input("x_min", value=-1.0)
    x_max = col2.number_input("x_max", value=1.0)

    # create run button
    if st.button("Run"):
        model = NN(
            n_hidden_layer_neurons,
            img.shape[-1],
            activation,
            kernel_first_layer_init,
            kernel_other_layers_init,
            bias_first_layer_init,
            bias_other_layers_init,
        )

        x1, x2 = np.meshgrid(
            np.linspace(x_min, x_max, img.shape[0]),
            np.linspace(x_min, x_max, img.shape[1]),
        )
        train_x = np.stack([x1.flatten(), x2.flatten()], axis=-1)
        train_y = np.array(img).reshape(-1, img.shape[-1])

        params, losses, outs = fit(
            key,
            model,
            train_x,
            train_y,
            lr,
            batch_size,
            n_iters,
        )
        st.line_chart(losses)
