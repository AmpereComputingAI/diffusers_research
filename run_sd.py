import hashlib
import traceback

import torch
from diffusers import StableDiffusion3Pipeline

DEBUG = False


class Condition:
    def __init__(self, condition: str, operands: dict):
        self.condition = condition
        self.operands = {key: Data(value) for key, value in operands.items()}
        self.location = traceback.extract_stack()[-3]


class Loop:
    def __init__(self, body, operands):
        self.body = body
        self.operands = {key: Data(value) for key, value in operands.items()}
        self.location = traceback.extract_stack()[-3]
        self.iter = 0

    def iteration(self):
        self.iter += 1


def hash_tensor(tensor):
    tensor = tensor.clone().detach()
    tensor_bytes = tensor.cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()


class Data:
    def __init__(self, value):
        if isinstance(value, list):
            self.type = list
            self.hash = None
            self.meta = {"dims": None}
        elif value is None:
            self.type = None
            self.hash = None
            self.meta = None
        elif isinstance(value, bool):
            self.type = bool
            self.hash = None
            self.meta = {"value": value}
        elif isinstance(value, str):
            self.type = str
            self.hash = None
            self.meta = {"size": len(value)}
        elif isinstance(value, torch.Tensor):
            self.type = torch.Tensor
            self.hash = hash_tensor(value)
            self.meta = {"shape": value.shape, "dtype": value.dtype, "device": value.device}
        else:
            self.type = "PythonClass"
            self.hash = None
            self.meta = None
            print(f"!!! {type(value)} !!!")


class Op:
    name = None
    def __init__(self, inp, out, condition_stack, loop_stack):
        self.location = traceback.extract_stack()[-4]
        self.input = {key: Data(value) for key, value in inp.items()}
        self.output = {key: Data(value) for key, value in out.items()}
        self.condition_stack = [condition for condition in condition_stack]
        self.loop_stack = [loop for loop in loop_stack]


class TorchTensor(Op):
    name = "torch.tensor"
    def __init__(self, inp, out, condition_stack, loop_stack):
        super().__init__(inp, out, condition_stack, loop_stack)


class IsTensor(Op):
    name = "torch.is_tensor"
    def __init__(self, inp, out, condition_stack, loop_stack):
        super().__init__(inp, out, condition_stack, loop_stack)


ops = {op.name: op for op in [
    TorchTensor,
    IsTensor
]}


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

    def add_op(self, name, inp, out):
        self.ops.append(ops[name](inp, out, self.condition_stack, self.loop_stack))

    def vomit(self):
        raise NotImplementedError("tracing not implemented")

    def debug(self, text=""):
        if DEBUG:
            print(f"DEBUG: {traceback.extract_stack()[-2]} [{text}]")

    def summary(self):
        for op in self.ops:
            print("------")
            print(op.name)
            print(op.location)
            print({key: [val.type, val.hash, val.meta] for key, val in op.input.items()})
            print({key: [val.type, val.hash, val.meta] for key, val in op.output.items()})
            print("Conditions:")
            for cond in op.condition_stack:
                print(cond.condition)
            print("Loops:")
            for loop in op.loop_stack:
                print(loop.body, loop.iter)


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


if __name__ == "__main__":
    main()
