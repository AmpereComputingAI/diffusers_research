import traceback
from email.quoprimime import body_encode

import torch
from diffusers import StableDiffusion3Pipeline


class Condition:
    def __init__(self, condition: str, operands: dict):
        self.condition = condition
        self.operands = operands
        self.location = traceback.extract_stack()[-3]


class Loop:
    def __init__(self, body, operands):
        self.body = body
        self.operands = operands
        self.iter = 0

    def iter(self):
        self.iter += 1


class Tracer:
    def __init__(self):
        self.ops = []
        self.condition_stack = []
        self.loop_stack = []

    def add_loop(self, body: str, operands: dict):
        self.loop_stack.append(Loop(body, operands))
        return len(self.loop_stack) - 1

    def reset_loop_stack(self, idx: int):
        self.loop_stack = self.loop_stack[:idx]

    def add_condition(self, condition: str, operands: dict):
        self.condition_stack.append(Condition(condition, operands))
        return len(self.condition_stack) - 1

    def reset_condition_stack(self, idx: int):
        self.condition_stack = self.condition_stack[:idx]

    def add_op(self, name, inp, output):
        print(name)
        print(inp)
        print(output)
        for cond in self.condition_stack:
            print(cond)
        for loop in self.loop_stack:
            print(loop)
        sd

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


if __name__ == "__main__":
    main()
