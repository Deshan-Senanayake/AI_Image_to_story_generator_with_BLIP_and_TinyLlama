import gradio as gr
from PIL import Image
from blip_caption import generate_caption

def caption_from_image(image, tone, mode):
    if image is None:
        return "Please upload an image."
    caption = generate_caption(image)
    print("Generated caption:", caption)
    return caption



iface = gr.Interface(
    fn=caption_from_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🖼️ Image Captioning with BLIP",
    description="Upload an image to generate a caption using the BLIP model."
)

if __name__ == "__main__":
    iface.launch()
