from .layers import *
from .helpers import *
from .generic import *
from .trajectory_dfm import *


def build_context(model, dataset, input_dict):
    context = None
    if getattr(model, "context_model", None) is not None:
        context = dict()
        if getattr(dataset, "variable_environment", False):
            env_normalized = input_dict[f"{dataset.field_key_env}_normalized"]
            context["env"] = env_normalized
        task_normalized = input_dict[f"{dataset.field_key_task}_normalized"]
        context["tasks"] = task_normalized
    return context


try:
    from .diffusion_models import *
except ModuleNotFoundError:
    # DFM-first paths do not require the legacy diffusion stack at import time.
    pass
