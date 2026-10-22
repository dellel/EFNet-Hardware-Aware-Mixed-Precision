import torch
import torch.nn as nn
from torch.func import functional_call

from quantization.learnable_quantization import LearnableWeightQuantizer


class MixedPrecisionTemporalEFNet(nn.Module):
    """
    Learnable weight precision per component.
    Activations and biases remain in floating point.
    The original FP32 parameters are preserved.
    """

    def __init__(self, base_model, initial_bits=8):
        super().__init__()

        self.base_model = base_model

        # One shared learnable precision distribution per component.
        self.quantizers = nn.ModuleDict({
            "encoder": LearnableWeightQuantizer(initial_bits=initial_bits),
            "temporal": LearnableWeightQuantizer(initial_bits=initial_bits),
            "regressor": LearnableWeightQuantizer(initial_bits=initial_bits),
        })

        self.weight_to_component = {}

        for name, module in self.base_model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue

            if name.startswith("encoder."):
                component = "encoder"
            elif name.startswith("temporal."):
                component = "temporal"
            elif name.startswith("regressor."):
                component = "regressor"
            else:
                continue

            self.weight_to_component[f"{name}.weight"] = component

    def forward(self, images):
        # Start with the original parameters.
        parameters = dict(self.base_model.named_parameters())
        parameters.update(dict(self.base_model.named_buffers()))

        # Replace only the effective weights used in this forward pass.
        # Original FP32 parameters remain unchanged.
        for parameter_name, component in self.weight_to_component.items():
            original_weight = parameters[parameter_name]
            quantizer = self.quantizers[component]
            parameters[parameter_name] = quantizer(original_weight)

        return functional_call(
            self.base_model,
            parameters,
            (images,),
            strict=True,
        )

    def precision_penalty(self):
        penalties = [
            quantizer.precision_penalty()
            for quantizer in self.quantizers.values()
        ]
        return torch.stack(penalties).mean()

    def expected_precisions(self):
        return {
            name: quantizer.expected_bits().item()
            for name, quantizer in self.quantizers.items()
        }

    def selected_precisions(self):
        return {
            name: quantizer.selected_bit_width()
            for name, quantizer in self.quantizers.items()
        }
