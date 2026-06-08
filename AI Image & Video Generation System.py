```python
# AI Image Generation using Stable Diffusion

import torch
from diffusers import StableDiffusionPipeline

# Device Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using Device:", device)

# Load Model
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

pipe = pipe.to(device)

# Memory Optimization
pipe.enable_attention_slicing()

print("Model Loaded Successfully")

# User Prompt
user_prompt = input("Enter your prompt: ")

# Enhanced Prompt
final_prompt = f"""
{user_prompt},
ultra realistic,
8k,
cinematic lighting,
highly detailed,
sharp focus,
professional photography,
realistic skin texture,
depth of field,
movie quality
"""

# Generate Image
image = pipe(
    final_prompt,
    height=768,
    width=768,
    num_inference_steps=50,
    guidance_scale=9
).images[0]

# Save Image
image.save("generated_image.png")

print("Image Saved Successfully")
```
