import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableWeightQuantizer(nn.Module):
    """
    Differentiable mixed-precision weight quantization.

    Candidate precisions: 2 to 8 bits.
    A learnable distribution assigns importance to each precision.
    The forward pass combines quantized weights using soft probabilities.
    """

    def __init__(self, initial_bits=8, min_bits=2, max_bits=8, temperature=1.0):
        super().__init__()

        if not min_bits <= initial_bits <= max_bits:
            raise ValueError("Invalid bit-width range.")

        self.min_bits = min_bits
        self.max_bits = max_bits
        self.temperature = temperature

        # Learnable logits over candidate bit-widths.
        self.bit_logits = nn.Parameter(
            torch.zeros(max_bits - min_bits + 1)
        )

        # Initialize the distribution toward initial_bits.
        with torch.no_grad():
            self.bit_logits[initial_bits - min_bits] = 4.0

    def bit_probabilities(self):
        return F.softmax(self.bit_logits / self.temperature, dim=0)

    def expected_bits(self):
        bits = torch.arange(
            self.min_bits,
            self.max_bits + 1,
            device=self.bit_logits.device,
            dtype=self.bit_logits.dtype,
        )
        return torch.sum(self.bit_probabilities() * bits)

    def forward(self, weights):
        probabilities = self.bit_probabilities()
        quantized_candidates = []

        for bits in range(self.min_bits, self.max_bits + 1):
            qmax = 2 ** (bits - 1) - 1

            scale = (
                weights.detach().abs().max().clamp_min(1e-8) / qmax
            )

            quantized = (
                torch.round(weights / scale)
                .clamp(-qmax, qmax)
                * scale
            )

            # Straight-through estimator for the weight gradient.
            quantized_ste = weights + (quantized - weights).detach()
            quantized_candidates.append(quantized_ste)

        # Differentiable mixture: gradients reach the precision logits.
        output = torch.zeros_like(weights)

        for index, quantized in enumerate(quantized_candidates):
            output = output + probabilities[index] * quantized

        return output

    def precision_penalty(self):
        """Expected precision normalized to approximately [0, 1]."""
        return (
            self.expected_bits() - self.min_bits
        ) / (self.max_bits - self.min_bits)

    def selected_bit_width(self):
        """Most probable precision, suitable for discrete selection."""
        index = torch.argmax(self.bit_probabilities()).item()
        return self.min_bits + index
