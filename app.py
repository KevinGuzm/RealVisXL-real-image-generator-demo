import torch
from diffusers import DiffusionPipeline
import gradio as gr

# Verify if CUDA is available
if torch.cuda.is_available():
    device = 'cuda'
    dtype = torch.float16
else:
    device = 'cpu'
    dtype = torch.float32

# Load the pipeline
pipeline = DiffusionPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=dtype
)
pipeline.to(device)

def generate_image(prompt, negative_prompt):
    try:
        image = pipeline(prompt, negative_prompt=negative_prompt).images[0]
        return image
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface
with gr.Blocks() as app:
    gr.Markdown("## Image Generation with Diffusers")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
    with gr.Row():
        negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter your negative prompt here...")
    generate_button = gr.Button("Generate")
    with gr.Row():
        image_output = gr.Image(label="Generated Image")

    generate_button.click(
        generate_image,
        inputs=[prompt, negative_prompt],
        outputs=image_output
    )

app.launch(share=True)
