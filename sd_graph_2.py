import logging
from typing import Any, Callable, Dict, List, Optional, Union


class Variable:
    def __init__(self, name, dtype, description=""):
        self.name = name
        self.dtype = dtype
        self.description = description


class Input(Variable):
    def __init__(self, name, dtype, description, required=True):
        super().__init__(name, dtype, description)
        self.required = required


class Op:
    def __init__(self, caller, info=""):
        self.section = caller.__class__.__name__
        if info != "":
            logging.info(info)


class PythonBuiltin(Op):
    def __init__(self, caller, info=""):
        super().__init__(caller, info)


class Conditional(PythonBuiltin):
    def __init__(self, sd3, caller, info=""):
        super().__init__(caller, info)
        sd3.add_op(self)


class Assignment(PythonBuiltin):
    def __init__(self, sd3, caller, info=""):
        super().__init__(caller, info)
        sd3.add_op(self)


class Mul(PythonBuiltin):
    def __init__(self, sd3, caller, info=""):
        super().__init__(caller, info)
        sd3.add_op(self)


class Section:
    def __init__(self, sd3, obj, purpose):
        sd3.register_section(obj.__class__.__name__, purpose)
        logging.info(f"\n> {purpose}")


class GetHW(Section):
    def __init__(self, sd3, height, width):
        super().__init__(sd3, self, "Figuring out target image height and width")

        Conditional(sd3, self, "Checking if image height was provided as argument")
        if height is not None:
            Assignment(sd3, self, f"Using supplied height: {height}")
        else:
            Mul(sd3, self, "Deriving height using formula: 'self.default_sample_size * self.vae_scale_factor'")
            Assignment(sd3, self)

        Conditional(sd3, self, "Checking if image width was provided as argument")
        if width is not None:
            Assignment(sd3, self, f"Using supplied width: {width}")
        else:
            Mul(sd3, self, "Deriving width using formula: 'self.default_sample_size * self.vae_scale_factor'")
            Assignment(sd3, self)


class GetBS(Section):
    def __init__(self, sd3, prompt, prompt_embeds):
        super().__init__(sd3, self, "Figuring out batch size")




class SD3:
    def __init__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None
    ):
        self.ops = []
        self.sections = {}
        logging.info("> Loading model 'stabilityai/stable-diffusion-3.5-large' with torch_dtype=torch.bfloat16")
        # check inputs with original function
        GetHW(self, height, width)

    def register_section(self, name, purpose):
        if name not in self.sections.keys():
            self.sections[name] = purpose

    def add_op(self, op):
        self.ops.append(op)





def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    trace = SD3("hyrax holding beer")
    print(trace.ops)
    print(trace.sections)


if __name__ == "__main__":
    main()
