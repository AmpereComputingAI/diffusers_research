import hashlib
import traceback

import torch
from diffusers import StableDiffusion3Pipeline

DEBUG = False


# class Condition:
#     def __init__(self, condition: str, operands: dict):
#         self.condition = condition
#         self.operands = {key: Data(value) for key, value in operands.items()}
#         self.location = traceback.extract_stack()[-3]
#
#
# class Loop:
#     def __init__(self, body, operands):
#         self.body = body
#         self.operands = {key: Data(value) for key, value in operands.items()}
#         self.location = traceback.extract_stack()[-3]
#         self.iter = 0
#
#     def iteration(self):
#         self.iter += 1


class Section:
    def __init__(self, annotation: str):
        self.annotation = annotation


def hash_tensor(tensor):
    tensor = tensor.clone().detach()
    tensor_bytes = tensor.to(dtype=torch.float32).cpu().contiguous().numpy().tobytes()
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
        elif isinstance(value, int):
            self.type = int
            self.hash = None
            self.meta = {"value": value}
        elif isinstance(value, float):
            self.type = float
            self.hash = None
            self.meta = {"value": value}
        elif isinstance(value, tuple):
            self.type = tuple
            self.hash = None
            self.meta = None
        elif isinstance(value, torch.Size):
            self.type = torch.Size
            self.hash = None
            self.meta = {"value": value}
        elif isinstance(value, torch.dtype):
            self.type = torch.dtype
            self.hash = None
            self.meta = {"value": value}
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
    def __init__(self, args, inp, out, section_stack):
        self.location = traceback.extract_stack()[-4]
        self.args = {key: Data(value) for key, value in args.items()}
        self.input = {key: Data(value) for key, value in inp.items()}
        self.output = {key: Data(value) for key, value in out.items()}
        self.section_stack = [section for section in section_stack]
        # self.condition_stack = [condition for condition in condition_stack]
        # self.loop_stack = [loop for loop in loop_stack]


class Tensor(Op):
    name = "torch.tensor"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class IsTensor(Op):
    name = "torch.is_tensor"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class TensorSize(Op):
    name = "torch.Tensor.size"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class TensorTo(Op):
    name = "torch.Tensor.to"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class TensorView(Op):
    name = "torch.Tensor.view"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class nnEmbedding(Op):
    name = "torch.nn.Embedding"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class Add(Op):
    name = "torch.add"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class Sub(Op):
    name = "torch.sub"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class TensorMaskedFill_(Op):
    name = "torch.Tensor.masked_fill_"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class Lt(Op):
    name = "torch.lt"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class Arange(Op):
    name = "torch.arange"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class Full(Op):
    name = "torch.full"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class Zeros(Op):
    name = "torch.zeros"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)

class Cat(Op):
    name = "torch.cat"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)

class OnesLike(Op):
    name = "torch.ones_like"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)

class TensorExpand(Op):
    name = "torch.Tensor.expand"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class Tril(Op):
    name = "torch.tril"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class Linear(Op):
    name = "torch.nn.Linear"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class LayerNorm(Op):
    name = "torch.nn.LayerNorm"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class SDPA(Op):
    name = "torch.nn.functional.scaled_dot_product_attention"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class TensorTranspose(Op):
    name = "torch.Tensor.transpose"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class TensorReshape(Op):
    name = "torch.Tensor.reshape"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


ops = {op.name: op for op in [
    Tensor,
    IsTensor,
    TensorSize,
    TensorView,
    nnEmbedding,
    Add,
    Sub,
    TensorMaskedFill_,
    Lt,
    Arange,
    Full,
    Zeros,
    Cat,
    OnesLike,
    Tril,
    TensorExpand,
    TensorTo,
    LayerNorm,
    Linear,
    TensorTranspose,
    TensorReshape,
    SDPA
]}


class Tracer:
    def __init__(self):
        self.ops = []
        self.section_stack = []
        # self.condition_stack = []
        # self.loop_stack = []

    class _Section:
        def __init__(self, tracer, annotation):
            self.tracer = tracer
            self.annotation = annotation

        def __enter__(self):
            self.tracer.section_stack.append(Section(self.annotation))

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.tracer.section_stack.pop()

    def get_dict(self, items):
        return {f"{i}": val for i, val in enumerate(items)}

    def section(self, annotation):
        return self._Section(self, annotation)

    def add_loop(self, body, operands):
        pass

    def reset_loop_stack(self, _):
        pass

    def add_condition(self, condition, operands):
        pass

    def reset_condition_stack(self, _):
        pass

    # def add_loop(self, body: str, operands: dict):
    #     self.loop_stack.append(Loop(body, operands))
    #     return len(self.loop_stack) - 1
    #
    # def reset_loop_stack(self, idx: int):
    #     self.loop_stack = self.loop_stack[:idx]
    #
    # def add_condition(self, condition: str, operands: dict):
    #     self.condition_stack.append(Condition(condition, operands))
    #     return len(self.condition_stack) - 1
    #
    # def reset_condition_stack(self, idx: int):
    #     self.condition_stack = self.condition_stack[:idx]

    def add_op(self, name, inp, out, args=None):
        if args is None:
            args = {}
        self.ops.append(ops[name](args, inp, out, self.section_stack))

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
            print("Section:")
            print(" -> ".join([str(section.annotation)[:20] for section in op.section_stack]))
            # print("Conditions:")
            # for cond in op.condition_stack:
            #     print(cond.condition)
            # print("Loops:")
            # for loop in op.loop_stack:
            #     print(loop.body, loop.iter)
            #     print({key: [val.type, val.hash, val.meta] for key, val in loop.operands.items()})


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
