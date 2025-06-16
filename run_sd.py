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
    tensor_data_ptr = tensor.storage().data_ptr()
    tensor = tensor.clone().detach()
    tensor_bytes = tensor.to(dtype=torch.float32).cpu().contiguous().numpy().tobytes()
    return str(tensor_data_ptr) + "_" + hashlib.sha256(tensor_bytes).hexdigest()


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
                op.dependants[self.hash] = set()
        elif isinstance(value, torch.device):
            self.meta = {"value": value}
        else:
            assert False, type(value)


class Op:
    name = None
    def __init__(self, args, inp, out, section_stack):
        self.dependencies = set()
        self.dependants = {}
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
        assert len(self.input_tensors) == 0


class IsTensor(Op):
    name = "torch.is_tensor"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1


class TensorSize(Op):
    name = "torch.Tensor.size"
    def __init__(self, args, inp, out, section):
        assert "input" in inp
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1


class TensorTo(Op):
    name = "torch.Tensor.to"
    def __init__(self, args, inp, out, section):
        assert "input" in inp
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1


class TensorView(Op):
    name = "torch.Tensor.view"
    def __init__(self, args, inp, out, section):
        assert "input" in inp
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1


class nnEmbedding(Op):
    name = "torch.nn.Embedding"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1


class Add(Op):
    name = "torch.add"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 2, self.location


class Sub(Op):
    name = "torch.sub"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 2, self.location


class Mul(Op):
    name = "torch.mul"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 2, self.location


class TensorContiguous(Op):
    name = "torch.Tensor.contiguous"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class TensorMaskedFill(Op):
    name = "torch.Tensor.masked_fill"
    def __init__(self, args, inp, out, section):
        assert "input" in inp
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class TensorArgmax(Op):
    name = "torch.Tensor.argmax"
    def __init__(self, args, inp, out, section):
        assert "input" in inp
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class Lt(Op):
    name = "torch.lt"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) >= 1, self.location


class Arange(Op):
    name = "torch.arange"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 0, self.location


class Full(Op):
    name = "torch.full"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 0, self.location


class Zeros(Op):
    name = "torch.zeros"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 0, self.location

class Ones(Op):
    name = "torch.ones"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 0, self.location

class Cat(Op):
    name = "torch.cat"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) >= 2, self.location

class OnesLike(Op):
    name = "torch.ones_like"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location

class ZerosLike(Op):
    name = "torch.zeros_like"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location

class FullLike(Op):
    name = "torch.full_like"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location

class Min(Op):
    name = "torch.min"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location

class Where(Op):
    name = "torch.where"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) >= 1, self.location

class TensorExpand(Op):
    name = "torch.Tensor.expand"

    def __init__(self, args, inp, out, section):
        assert "input" in inp
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class TensorSelection(Op):
    name = "torch.Tensor.selection"

    def __init__(self, args, inp, out, section):
        assert "input" in inp
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class Tril(Op):
    name = "torch.tril"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class Pow(Op):
    name = "torch.pow"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class Div(Op):
    name = "torch.div"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 2, self.location


class Log(Op):
    name = "torch.log"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class Mean(Op):
    name = "torch.mean"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class RSqrt(Op):
    name = "torch.rsqrt"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class Linear(Op):
    name = "torch.nn.Linear"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class LayerNorm(Op):
    name = "torch.nn.LayerNorm"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class SDPA(Op):
    name = "torch.nn.functional.scaled_dot_product_attention"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) >= 3, self.location


class TensorTranspose(Op):
    name = "torch.Tensor.transpose"
    def __init__(self, args, inp, out, section):
        assert "input" in inp
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class TensorReshape(Op):
    name = "torch.Tensor.reshape"
    def __init__(self, args, inp, out, section):
        assert "input" in inp
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class TensorRepeat(Op):
    name = "torch.Tensor.repeat"
    def __init__(self, args, inp, out, section):
        assert "input" in inp
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class TensorUnsqueeze(Op):
    name = "torch.Tensor.unsqueeze"
    def __init__(self, args, inp, out, section):
        assert "input" in inp
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class Sigmoid(Op):
    name = "torch.sigmoid"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class Abs(Op):
    name = "torch.abs"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class Gelu(Op):
    name = "torch.nn.functional.gelu"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class ApexFusedRMSNorm(Op):
    name = "apex.normalization.FusedRMSNorm"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)


class MatMul(Op):
    name = "torch.matmul"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 2, self.location


class TensorPermute(Op):
    name = "torch.Tensor.permute"
    def __init__(self, args, inp, out, section):
        assert "input" in inp
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class SoftMax(Op):
    name = "torch.nn.functional.softmax"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class Tanh(Op):
    name = "torch.tanh"
    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class TensorTypeAs(Op):
    name = "torch.Tensor.type_as"
    def __init__(self, args, inp, out, section):
        assert "input" in inp
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


class Pad(Op):
    name = "torch.nn.functional.pad"

    def __init__(self, args, inp, out, section):
        super().__init__(args, inp, out, section)
        assert len(self.input_tensors) == 1, self.location


ops = {op.name: op for op in [
    Tensor,
    IsTensor,
    TensorSize,
    TensorView,
    nnEmbedding,
    Add,
    Sub,
    Mul,
    TensorMaskedFill,
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
    TensorSelection,
    TensorContiguous
]}


# class Graph:
#     def __init__(self, ops):
#         self.ops = ops
#         self.vars = {}
#
#     class Variable:
#         codes = [str(i) for i in range(10000)]
#         #codes = list(string.ascii_uppercase)
#         occupied_codes = []
#
#         def __init__(self, location, dependants):
#             self.latest_call = location
#             self.dependants = [dep for dep in dependants]
#             self.code = None
#             self.get_code()
#
#         def get_code(self):
#             for code in self.codes:
#                 if code not in self.occupied_codes:
#                     self.code = code
#                     self.occupied_codes.append(code)
#                     break
#             else:
#                 assert False
#
#         def free_code(self, code):
#             self.occupied_codes.remove(code)
#
#         def add_dependants(self, dependants):
#             self.dependants += [dep for dep in dependants]
#
#         def remove_dependant(self, caller):
#             self.dependants.remove(caller)
#             if len(self.dependants) == 0:
#                 self.free_code(self.code)
#
#     def new_var(self, tensor_hash, op):
#         # if tensor_hash in self.vars.keys():
#         #     self.vars[tensor_hash].add_dependants(op.dependants[tensor_hash])
#         #     self.vars[tensor_hash].latest_call = op.location
#         assert tensor_hash not in self.vars
#         if len(op.dependants[tensor_hash]) > 0:
#             self.vars[tensor_hash] = self.Variable(op.location, op.dependants[tensor_hash])
#         else:
#             return "_"
#         return self.vars[tensor_hash].code
#
#     def get_var(self, tensor_hash, caller):
#         try:
#             self.vars[tensor_hash].remove_dependant(caller)
#         except ValueError as e:
#             print(caller.location)
#             print(self.vars[tensor_hash].code)
#             print(self.vars[tensor_hash].dependants)
#             print(caller)
#             raise e
#         code = self.vars[tensor_hash].code
#         if len(self.vars[tensor_hash].dependants) == 0:
#             self.vars.pop(tensor_hash)
#         return code
#
#     def print(self):
#         print("\nGraph:\n")
#         processed = []
#         while len(self.ops) - len(processed) > 0:
#             for op in self.ops:
#                 if all([dep in processed for dep in op.dependencies]):
#                     # print(op.name)
#                     # print(op.location)
#                     # print(self.vars)
#                     skip = False
#                     for tensor in op.output_tensors:
#                         if tensor in self.vars:
#                             if tensor not in op.input_tensors or len(self.vars[tensor].dependants) > 1:
#                                 skip = True
#                                 break
#                     if skip:
#                         continue
#                     inputs = ", ".join([self.get_var(tensor, op) for tensor in op.input_tensors])
#                     outputs = [self.new_var(tensor, op) for tensor in op.output_tensors]
#                     if len(outputs) > 0:
#                         outputs = ", ".join(outputs) + " = "
#                     else:
#                         outputs = ""
#                     print(f"{outputs}{op.__class__.__name__}({inputs}) [{op.location}]")
#                     # print(op.name)
#                     # print(op.location)
#                     # print(op.input_tensors)
#                     # print(op.output_tensors)
#                     processed.append(op)
#         print(f"\nVars left out: {len(self.vars)}")
#         for var in self.vars.values():
#             print(var.latest_call)


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
        Graph(self.ops, self.preloaded_tensors).print()


class Graph:
    def __init__(self, ops, preloaded_tensors):
        self.vars = {}
        self.ops = ops
        self.preloaded_tensors = preloaded_tensors
        tensor_map = {tensor.output_tensors[0]: tensor for tensor in self.preloaded_tensors}
        for op in self.ops:
            # print("------")
            # print(op.name)
            # print(op.location)
            # print({key: [val.type, val.hash, val.meta] for key, val in op.args.items()})
            # print({key: [val.type, val.hash, val.meta] for key, val in op.input.items()})
            # print({key: [val.type, val.hash, val.meta] for key, val in op.output.items()})
            # print(op.input_tensors)
            # print(op.output_tensors)
            # print(" -> ".join([str(section.annotation).split("(")[0] for section in op.section_stack]))

            for tensor in op.input_tensors:
                op.dependencies.add(tensor_map[tensor])
                tensor_map[tensor].dependants[tensor].add(op)

            for tensor in op.output_tensors:
                tensor_map[tensor] = op

            # print("------")
            # print(op.name)
            # print(op.location)
            # print("Deps:")
            # for dep in op.dependencies:
            #     print(dep.name)
            #     print(dep.location)

    class Variable:
        #codes = list(string.ascii_uppercase)
        codes = [str(i) for i in range(10000)]
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
            self.occupied_codes.remove(code)

        def add_dependants(self, dependants):
            self.dependants += [dep for dep in dependants]

        def remove_dependant(self, caller):
            self.dependants.remove(caller)
            if len(self.dependants) == 0:
                self.free_code(self.code)
                return True
            else:
                return False

    def get_var(self, tensor_hash, op):
        code = self.vars[tensor_hash].code
        if self.vars[tensor_hash].remove_dependant(op):
            self.vars.pop(tensor_hash)
        return code

    def new_var(self, tensor_hash, op):
        if tensor_hash in self.vars.keys():
            self.vars[tensor_hash].add_dependants(op.dependants[tensor_hash])
        else:
            self.vars[tensor_hash] = self.Variable(op.dependants[tensor_hash])
        return self.vars[tensor_hash].code

    def print(self):
        processed_ops = set()
        ops = self.ops.copy()
        while len(ops) > 0:
            for i, op in enumerate(ops):
                if all([dep in processed_ops for dep in op.dependencies]):
                    # skip = False
                    # for tensor_hash in op.output_tensors:
                    #     if tensor_hash in self.vars:
                    #         skip = True
                    # if skip:
                    #     continue
                    #print(op.location)
                    inputs = ", ".join([self.get_var(tensor_hash, op) for tensor_hash in op.input_tensors])

                    outputs = [self.new_var(tensor_hash, op) for tensor_hash in op.output_tensors]
                    if len(outputs) > 0:
                        outputs = ", ".join(outputs) + " = "
                    else:
                        outputs = ""

                    print(f"{outputs}{op.__class__.__name__}({inputs})")
                    processed_ops.add(ops.pop(i))
                    break


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
