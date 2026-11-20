import torch
from src.ctc_loss import CTCSequenceLoss


def test_ctc_loss():
    batch_size = 2
    time_steps = 20
    num_classes = 10

    logits = torch.randn(batch_size, time_steps, num_classes)

    targets = torch.tensor([1, 2, 3, 4, 5])

    input_lengths = torch.tensor([20, 20])
    target_lengths = torch.tensor([3, 2])

    criterion = CTCSequenceLoss(blank=0)

    loss = criterion(
        logits,
        targets,
        input_lengths,
        target_lengths
    )

    assert torch.isfinite(loss), "Loss is not finite!"
    assert loss.item() >= 0, "Loss should be non-negative!"

    print("CTC loss test passed!")
    print("Loss:", loss.item())


if __name__ == "__main__":
    test_ctc_loss()