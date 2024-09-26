from flask import Flask, render_template, request

import torch 
from diffusers import StableDiffusionPipeline

import base64 
from io import BytesIO

# Loading model
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Starting the Flask app
app = Flask(__name__)

@app.route('/')
def init():
    return render_template('home.html')

@app.route('/submit', methods=["POST"])
def generate_image():
    prompt = request.form["query"]  # Adjusted to match form field name
    print(f'Generating an image of: {prompt}')
    
    image = pipe(prompt).images[0]
    
    print("Image generated! Converting image ...")

    buffered = BytesIO()
    image.save(buffered, format="PNG")   
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    img_str = f"data:image/png;base64,{img_str}"

    print("Sending image ...")
    return render_template('home.html', que = prompt, generated_image=img_str)

if __name__ == '__main__':
    app.run(debug = True)