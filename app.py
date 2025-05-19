import gradio as gr
from fastai.vision.all import *
from PIL import Image, UnidentifiedImageError
import numpy as np
import io

# Load the model
learn = load_learner('ArdaBaran_ADA447_MidtermProject_PetBreedClassifier.pkl')


def classify_image(img):
    try:
      
        if img is None:
            return "No image received."

       
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert("RGB")

       
        elif isinstance(img, bytes):
            img = Image.open(io.BytesIO(img)).convert("RGB")


        elif not isinstance(img, Image.Image):
            return f"Unsupported image format: {type(img)}"

       
        pred, idx, probs = learn.predict(img)
        return dict(zip(learn.dls.vocab, map(float, probs)))

    except UnidentifiedImageError:
        return "Unsupported or corrupted image format."
    except Exception as e:
        return f"Prediction failed: {str(e)}"

# Gradio interface
app = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(label="Upload any image (jpg, png, bmp, tiff, webp, gif)"),
    outputs=gr.Label(num_top_classes=3),
    title="Pet Breed Classifier",
    description="Upload a pet image in any format to classify its breed."
)

app.launch()
