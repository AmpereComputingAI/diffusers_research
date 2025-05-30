import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    prompt="A hyrax holding a sign that reads Ampere!",
    negative_prompt="Correct anatomy, healthy",
    num_inference_steps=28,
    guidance_scale=7
).images[0]
image.save("hyrax.png")
