## using diffusers and creating an image generation webapp using flask and ngrok

from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok

import torch 
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

import base64 
from io import BytesIO

#loading model
model = "./path/realisticVisionV60B1_v60B1VAE.safetensors"
pipe = StableDiffusionPipeline.from_single_file(model)
pipe.to("cuda")

#staring our flask app
app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def init():
    return render_template("home.html")

@app.route('/submit', methods = "POST")
def generate_image():
    prompt = request.form["prompt-input"]
    print(f'Generating an image of {prompt}')
    scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012,
                                   beta_schedule="scaled_linear")
    image = pipe(
    prompt,
    scheduler=scheduler,
    num_inference_steps=30,
    guidance_scale=7.5,
    ).images[0]
    
    print("Image generated! Converting image ...")

    buffered = BytesIO()
    image.save(buffered, format = "PNG")   
    img_str = base64.b64encode(buffered.getvalue())
    img_str = "data:image/png;base64," + str(img_str)[2:-1]

    print("Sending image ...")
    return render_template('index.html', generated_image=img_str)

if __name__ == '__main__':
    app.run()