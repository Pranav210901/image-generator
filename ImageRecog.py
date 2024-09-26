## using diffusers and creating an image generation webapp using flask and ngrok

from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok

import torch 
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

import base64 
from io import BytesIO

#loading model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

#staring our flask app
app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def init():
    return render_template("home.html")

@app.route('/submit', methods = ["POST"])
def generate_image():
    prompt = request.form["prompt-input"]
    print(f'Generating an image of {prompt}')
    
    image = pipe(prompt).images[0]
    
    print("Image generated! Converting image ...")

    buffered = BytesIO()
    image.save(buffered, format = "PNG")   
    img_str = base64.b64encode(buffered.getvalue())
    img_str = "data:image/png;base64," + str(img_str)[2:-1]

    print("Sending image ...")
    return render_template('home.html', generated_image=img_str)

if __name__ == '__main__':
    app.run()
