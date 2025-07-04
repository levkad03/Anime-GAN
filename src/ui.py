import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

from generator import Generator
from utils import load_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
Z_DIM = 1
CHANNELS_IMG = 3
FEATURES_GEN = 64
CHECKPOINT_PATH = "models/generator_checkpoint.pth.tar"
IMAGE_SIZE = 64


@st.cache_resource
def load_generator():
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    load_checkpoint(checkpoint, gen)

    gen.eval()
    return gen


gen = load_generator()

st.title("🎨 Anime Face Generator (DCGAN)")
st.write("Click the button to generate a new anime face!")

if st.button("Generate Face"):
    with torch.no_grad():
        noise = torch.randn(1, Z_DIM, 1, 1).to(device)
        fake = gen(noise)
        fake_image = fake.squeeze(0).detach().cpu()

        # Denormalize and convert to PIL image
        transform = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
        fake_image = transform(fake_image).clamp(0, 1)
        ndarr = fake_image.mul(255).byte().permute(1, 2, 0).numpy()
        pil_img = Image.fromarray(ndarr)

        st.image(pil_img, caption="Generated Anime Face", use_container_width=False)
