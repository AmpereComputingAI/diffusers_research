from typing import List


class Op:
    name = None
    description = None
    def __init__(self):
        pass


class Variable:
    def __init__(self, dtype, description=None, name=None, value=None):
        self.name = name
        self.dtype = dtype
        self.description = description


class Graph:
    def __init__(self, name=None):
        self.self = None
        self.name = name
        self.variables = {}
        self.graph = []

    def add_variable(self, name, dtype, description):
        assert name not in self.variables.keys()
        self.variables[name] = Variable(name, dtype, description)

    def add_op(self, op):
        assert isinstance(op, Graph) or isinstance(op, Op), op
        self.graph.append(op)
        return op

    def __getitem__(self, item):
        return self.variables[item]


class Conditional(Op):
    name = "conditional"
    def __init__(self, condition: str, variables: List[Variable]):
        super().__init__()
        assert all([isinstance(var, Variable) for var in variables]) and type(variables) is list
        self.condition = condition
        self.variables = variables
        self.if_branch = None
        self.else_branch = None


class Assign(Op):
    name = "assign"
    def __init__(self, variable, value):
        super().__init__()
        assert isinstance(variable, Variable) and (isinstance(value, Variable) or isinstance(value, Op))
        self.variable = variable
        self.value = value


class PythonBuiltin(Op):
    name = "python built-in"
    def __init__(self, code: str, variables: List[Variable]):
        super().__init__()
        assert all([type(var) is Variable for var in variables]) and type(variables) is list
        self.code = code
        self.variables = variables


class ExternalMethod(Op):
    name = "external method"
    def __init__(self, code: str, variables: List[Variable]):
        super().__init__()
        assert all([isinstance(var, Variable) for var in variables]) and type(variables) is list
        self.code = code
        self.variables = variables


class TorchTensorShape(Op):
    name = "torch.Tensor.shape"
    description = "Returns the size of the self tensor. Alias for size."
    def __init__(self, arg: str, variable: Variable):
        super().__init__()
        assert type(variable) is Variable
        self.arg = arg
        self.variable = variable


def get_batch_size(g, prompt, prompt_embeds):
    g.add_variable(name="batch_size", dtype="int",
                   description="Batch size (multiple independent images to be generated in a single pass)")
    bs = g["batch_size"]

    g = Graph("get batch size")
    conditional_0 = Conditional("prompt is not None and isinstance(prompt, str)", [prompt])

    conditional_0.if_branch = Assign(bs, Variable(dtype="int", value="1"))

    else_branch = Graph()
    conditional_1 = Conditional("prompt is not None and isinstance(prompt, list)", [prompt])
    else_branch.add_op(conditional_1)
    conditional_0.else_branch = else_branch

    conditional_1.if_branch = Assign(bs, PythonBuiltin("len(prompt)", [prompt]))
    conditional_1.else_branch = Assign(bs, TorchTensorShape("[0]", prompt_embeds))

    g.add_op(conditional_0)

    return g


def get_lora_scale(g):
    g.add_variable(name="lora_scale", dtype="float",
                   description="A lora scale that will be applied to all LoRA layers of the text encoder if LoRA "
                               "layers are loaded.")
    lora_scale = g["lora_scale"]
    joint_attention_kwargs = g.self["_joint_attention_kwargs"]
    g = Graph("get_lora_scale")
    conditional = Conditional("self.joint_attention_kwargs is not None", [joint_attention_kwargs])

    conditional.if_branch = Assign(lora_scale, PythonBuiltin(
        "self.joint_attention_kwargs.get('scale', None)", [joint_attention_kwargs]))
    conditional.else_branch = Assign(lora_scale, Variable(dtype="None", value="None"))

    g.add_op(conditional)
    return g


def adjust_lora_scale(g):
    sub_g = Graph("dynamically adjust the LoRA scale")

    conditional = Conditional("self.text_encoder is not None and USE_PEFT_BACKEND", [g.self])
    conditional.if_branch = ExternalMethod("scale_lora_layers(self.text_encoder, lora_scale)",
                                           [g.self, g["lora_scale"]])
    sub_g.add_op(conditional)

    conditional = Conditional("self.text_encoder_2 is not None and USE_PEFT_BACKEND", [g.self])
    conditional.if_branch = ExternalMethod("scale_lora_layers(self.text_encoder2, lora_scale)",
                                           [g.self, g["lora_scale"]])
    sub_g.add_op(conditional)
    return sub_g


def get_clip_prompt_embeds_0(g, gg):
    sub_g = Graph("get clip prompt embeds 0")



    return sub_g


def get_positive_embeds(g):
    sub_g = Graph("get positive prompt embeds")

    conditional = sub_g.add_op(Conditional("prompt_2 is not None", [g["prompt_2"]]))
    sub_g.add_variable("prompt_2", "str or List[str]", None)
    conditional.if_branch = Assign(sub_g["prompt_2"], g["prompt_2"])
    conditional.else_branch = Assign(sub_g["prompt_2"], g["prompt"])

    conditional = sub_g.add_op(Conditional("isinstance(prompt_2, str)", [sub_g["prompt_2"]]))
    conditional.if_branch = Assign(sub_g["prompt_2"], PythonBuiltin("[prompt_2]", [sub_g["prompt_2"]]))
    conditional.else_branch = Assign(sub_g["prompt_2"], g["prompt_2"])

    conditional = sub_g.add_op(Conditional("prompt_3 is not None", [g["prompt_3"]]))
    sub_g.add_variable("prompt_3", "str or List[str]", None)
    conditional.if_branch = Assign(sub_g["prompt_3"], g["prompt_3"])
    conditional.else_branch = Assign(sub_g["prompt_3"], g["prompt"])

    conditional = sub_g.add_op(Conditional("isinstance(prompt_3, str)", [sub_g["prompt_3"]]))
    conditional.if_branch = Assign(sub_g["prompt_3"], PythonBuiltin("[prompt_3]", [sub_g["prompt_3"]]))
    conditional.else_branch = Assign(sub_g["prompt_3"], g["prompt_3"])

    sub_g.add_op(get_clip_prompt_embeds_0(g, sub_g))

    return sub_g


def encode_prompt(g):
    sub_g = Graph("encode prompt")

    conditional = Conditional("lora_scale is not None and isinstance(self, SD3LoraLoaderMixin)", [g["lora_scale"], g.self])
    conditional.if_branch = adjust_lora_scale(g)
    sub_g.add_op(conditional)

    conditional = sub_g.add_op(Conditional("isinstance(prompt, str)", [g["prompt"]]))
    sub_g.add_variable("prompt", "list[str]", None)
    conditional.if_branch = Assign(sub_g["prompt"], PythonBuiltin("[prompt]", [g["prompt"]]))
    conditional.else_branch = Assign(sub_g["prompt"], g["prompt"])

    conditional = sub_g.add_op(Conditional("prompt is not None", [sub_g["prompt"]]))
    sub_g.add_variable("batch_size", "int", None)
    conditional.if_branch = Assign(sub_g["batch_size"], PythonBuiltin("len(prompt)", [sub_g["prompt"]]))
    conditional.else_branch = Assign(sub_g["batch_size"], TorchTensorShape("[0]", g["prompt_embeds"]))

    conditional = sub_g.add_op(Conditional("prompt_embeds is None", [g["prompt_embeds"]]))
    conditional.if_branch = get_positive_embeds(g)

    return sub_g


class Self(Variable):
    def __init__(self, name: str, dtype):
        super().__init__(dtype, name)
        self.variables = {}

    def add_variable(self, name, dtype, description):
        assert name not in self.variables.keys()
        self.variables[name] = Variable(name, dtype, description)

    def __getitem__(self, item):
        return self.variables[item]


def SD3_5():
    g = Graph("SD 3.5")
    g.self = Self("StableDiffusion3Pipeline(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin, "
                  "SD3IPAdapterMixin)", "class")
    g.add_variable(name="prompt", dtype="str or List[str]",
                   description="The prompt or prompts to guide the image generation. If not defined, one has to pass "
                               "`prompt_embeds` instead.")
    g.add_variable(name="prompt_2", dtype="str or List[str]",
                   description="The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. "
                               "If not defined, `prompt` will be used instead")
    g.add_variable(name="prompt_3", dtype="str or List[str]",
                   description="The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. "
                               "If not defined, `prompt` will be used instead")
    g.add_variable(name="prompt_embeds", dtype="torch.FloatTensor",
                   description="Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* "
                               "prompt weighting. If not provided, text embeddings will be generated from `prompt` "
                               "input argument.")
    g.add_variable(name="joint_attention_kwargs", dtype="dict",
                   description="A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as "
                               "defined under `self.processor` in [diffusers.models.attention_processor]"
                               "(https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).")

    g.self.add_variable(name="_joint_attention_kwargs", dtype="dict", description=None)
    g.add_op(Assign(g.self["_joint_attention_kwargs"], g["joint_attention_kwargs"]))
    g.add_op(get_batch_size(g, g["prompt"], g["prompt_embeds"]))
    g.add_op(get_lora_scale(g))
    g.add_op(encode_prompt(g))

    print([obj.name for obj in g.graph])


if __name__ == "__main__":
    SD3_5()
