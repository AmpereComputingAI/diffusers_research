import torch
from diffusers import StableDiffusion3Pipeline


class Condition:
    def __init__(self, condition, operands):
        self.condition = condition
        self.operands = operands


class Tracer:
    def __init__(self):
        self.ops = []
        self.condition_stack = []

    def add_condition(self, condition, operands):
        self.condition_stack.append(Condition(condition, operands))
        return len(self.condition_stack)

    def reset_condition_stack(self, idx):
        self.condition_stack = self.condition_stack[:idx-1]

    def vomit(self):
        raise NotImplementedError("tracing not implemented")


def main():
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large",
                                                    torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    tracer = Tracer()

    image = pipe(
        tracer,
        prompt="A hyrax holding a sign that reads Ampere!",
        negative_prompt="Correct anatomy, healthy",
        num_inference_steps=28,
        guidance_scale=7
    ).images[0]
    image.save("hyrax.png")

    print(tracer.condition_stack)
