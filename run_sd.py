import hashlib
import string
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
    def __init__(self, op, value, is_input: bool):
        self.type = type(value)
        self.hash = None
        self.meta = None
        self.items = None

        if isinstance(value, list):
            self.items = [Data(op, val, is_input) for val in value]
        elif value is None:
            return
        elif isinstance(value, bool):
            self.meta = {"value": value}
        elif isinstance(value, str):
            self.meta = {"len": len(value)}
        elif isinstance(value, int):
            self.meta = {"value": value}
        elif isinstance(value, float):
            self.meta = {"value": value}
        elif isinstance(value, tuple):
            self.items = [Data(op, val, is_input) for val in value]
        elif isinstance(value, torch.Size):
            self.meta = {"value": value}
            assert False, value
        elif isinstance(value, torch.dtype):
            self.meta = {"value": value}
        elif isinstance(value, torch.Tensor):
            self.hash = hash_tensor(value)
            self.meta = {"shape": value.shape, "dtype": value.dtype, "device": value.device}
            if is_input:
                op.input_tensors.append(self.hash)
            else:
                op.output_tensors.append(self.hash)
        else:
            assert False, type(value)


class Op:
    name = None
    def __init__(self, args, inp, out, section_stack):
        self.dependencies = set()
        self.dependants = set()
        self.input_tensors = []
        self.output_tensors = []
        self.location = traceback.extract_stack()[-4]
        self.args = {key: Data(self, value, True) for key, value in args.items()}
        self.input = {key: Data(self, value, True) for key, value in inp.items()}
        self.output = {key: Data(self, value, False) for key, value in out.items()}
        self.section_stack = [section for section in section_stack]
        # self.condition_stack = [condition for condition in condition_stack]
        # self.loop_stack = [loop for loop in loop_stack]

class PreloadedTensor(Op):
    name = "preloaded_tensor"
    def __init__(self):
        super().__init__({}, {}, {}, [])

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


class Mul(Op):
    name = "torch.mul"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)



class TensorMaskedFill_(Op):
    name = "torch.Tensor.masked_fill_"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class TensorArgmax(Op):
    name = "torch.Tensor.argmax"
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

class Ones(Op):
    name = "torch.ones"

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

class ZerosLike(Op):
    name = "torch.zeros_like"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)

class FullLike(Op):
    name = "torch.full_like"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)

class Min(Op):
    name = "torch.min"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)

class Where(Op):
    name = "torch.where"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)

class TensorExpand(Op):
    name = "torch.Tensor.expand"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class TensorSelection(Op):
    name = "torch.Tensor.selection"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class Tril(Op):
    name = "torch.tril"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class Pow(Op):
    name = "torch.pow"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class Div(Op):
    name = "torch.div"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class Log(Op):
    name = "torch.log"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class Mean(Op):
    name = "torch.mean"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class RSqrt(Op):
    name = "torch.rsqrt"

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


class TensorRepeat(Op):
    name = "torch.Tensor.repeat"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class TensorUnsqueeze(Op):
    name = "torch.Tensor.unsqueeze"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class Sigmoid(Op):
    name = "torch.sigmoid"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class Abs(Op):
    name = "torch.abs"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class Gelu(Op):
    name = "torch.nn.functional.gelu"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class ApexFusedRMSNorm(Op):
    name = "apex.normalization.FusedRMSNorm"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class MatMul(Op):
    name = "torch.matmul"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class TensorPermute(Op):
    name = "torch.Tensor.permute"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class SoftMax(Op):
    name = "torch.nn.functional.softmax"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class Tanh(Op):
    name = "torch.tanh"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class TensorTypeAs(Op):
    name = "torch.Tensor.type_as"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class Pad(Op):
    name = "torch.nn.functional.pad"

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
    Mul,
    TensorMaskedFill_,
    Lt,
    Arange,
    Full,
    Ones,
    Zeros,
    Cat,
    OnesLike,
    Tril,
    TensorExpand,
    TensorRepeat,
    TensorTo,
    LayerNorm,
    Linear,
    TensorTranspose,
    TensorReshape,
    TensorUnsqueeze,
    SDPA,
    Sigmoid,
    TensorArgmax,
    Gelu,
    Pow,
    Mean,
    RSqrt,
    ApexFusedRMSNorm,
    MatMul,
    TensorPermute,
    Abs,
    ZerosLike,
    Min,
    Where,
    FullLike,
    Div,
    Log,
    TensorTypeAs,
    SoftMax,
    Tanh,
    Pad,
    TensorSelection
]}


class Graph:
    def __init__(self, ops):
        self.ops = ops
        self.vars = {}

    class Variable:
        codes = list(string.ascii_uppercase)
        occupied_codes = []

        def __init__(self, dependants):
            self.dependants = [dep for dep in dependants]
            self.code = None
            self.get_code()

        def get_code(self):
            for code in self.codes:
                if code not in self.occupied_codes:
                    self.code = code
                    self.occupied_codes.append(code)
                    break
            else:
                assert False

        def free_code(self, code):
            self.occupied_codes.pop(code)

        def remove_dependant(self, caller):
            self.dependants.pop(caller)
            if len(self.dependants) == 0:
                self.free_code(self.code)

    def new_var(self, tensor_hash, dependants):
        assert tensor_hash not in self.vars.keys()
        self.vars[tensor_hash] = self.Variable(dependants)
        return self.vars[tensor_hash].code

    def get_var(self, tensor_hash, caller):
        self.vars[tensor_hash].remove_dependant(caller)
        return self.vars[tensor_hash].code

    def print(self):
        print("\nGraph:\n")
        processed = []
        while len(self.ops) - len(processed) > 0:
            for op in self.ops:
                if all([dep in processed for dep in op.dependencies]):
                    outputs = [self.new_var(tensor, op.dependants) for tensor in op.output_tensors]
                    if len(outputs) > 0:
                        outputs = ", ".join(outputs) + " = "
                    else:
                        outputs = ""
                    inputs = ", ".join([self.get_var(tensor, op) for tensor in op.input_tensors])
                    print(f"{outputs}{op.__class__.__name__}({inputs})")
                    # print(op.name)
                    # print(op.location)
                    # print(op.input_tensors)
                    # print(op.output_tensors)
                    processed.append(op)


class Tracer:
    def __init__(self):
        self.preloaded_tensors = []
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

    def add_preloaded_tensor(self, tensor):
        t = PreloadedTensor()
        Data(t, tensor, False)
        self.preloaded_tensors.append(t)

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
        tensor_map = {tensor.output_tensors[0]: tensor for tensor in self.preloaded_tensors}
        for op in self.ops:
            print("------")
            print(op.name)
            print(op.location)
            print({key: [val.type, val.hash, val.meta] for key, val in op.args.items()})
            print({key: [val.type, val.hash, val.meta] for key, val in op.input.items()})
            print({key: [val.type, val.hash, val.meta] for key, val in op.output.items()})
            print(op.input_tensors)
            print(op.output_tensors)
            print(" -> ".join([str(section.annotation).split("(")[0] for section in op.section_stack]))

            if len(op.input_tensors) > 0:
                for tensor in op.input_tensors:
                    op.dependencies.add(tensor_map[tensor])
                    tensor_map[tensor].dependants.add(op)
            if len(op.output_tensors) > 0:
                for tensor in op.output_tensors:
                    tensor_map[tensor] = op

        Graph(self.ops).print()


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
