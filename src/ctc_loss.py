
import torch
import torch.nn as nn


class CTCSequenceLoss(nn.Module):
    """
    Computes Connectionist Temporal Classification (CTC) loss
    for sheet-music symbol recognition.
    """

    def __init__(self, blank=0):
        super().__init__()

        self.ctc_loss = nn.CTCLoss(
            blank=blank,
            reduction="mean",
            zero_infinity=True
        )

    def forward(self, logits, targets, input_lengths, target_lengths):
        """
        Args:
            logits:         Model predictions (B, T, C)
            targets:        Concatenated target token IDs
            input_lengths:  Valid output timesteps per image
            target_lengths: Number of target symbols per image

        Returns:
            CTC loss value
        """

        # Convert raw model scores into log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        # PyTorch CTC expects dimensions (T, B, C)
        log_probs = log_probs.transpose(0, 1)

        # Calculate CTC loss
        loss = self.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths
        )

        return loss


###draft 
