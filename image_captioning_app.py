import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration,BlipProcessor

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):
    # Convert numpy array to PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')
    
    # Process the image
    inputs = processor(images=raw_image, return_tensors="pt")
    outputs = model.generate(**inputs)
    
    # Generate a caption for the image
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    # Decode the generated tokens to text and store it into `caption`
    return caption

iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="Image Captioning",
    description="This is a simple web app for generating captions for images using a trained model."
)

iface.launch()