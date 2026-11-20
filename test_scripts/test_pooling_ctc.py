import torch
from src.model import MaxPoolingLayer
from src.ctc_loss import CTCSequenceLoss


def test_pooling_ctc():

    # Two grayscale images, padded to width 40
    images = torch.randn(2, 1, 64, 40)

    pooling = MaxPoolingLayer(kernel_size=2, stride=2)
    features = pooling(images)

    # Original widths: 32 and 40
    # After pooling: 16 and 20
    input_lengths = torch.tensor([16, 20])

    assert features.shape == (2, 1, 32, 20)

    # Simulate RNN/classifier predictions
    logits = torch.randn(2, 20, 10)

    targets = torch.tensor([1, 2, 3, 4, 5])
    target_lengths = torch.tensor([3, 2])

    criterion = CTCSequenceLoss(blank=0)

    loss = criterion(
        logits,
        targets,
        input_lengths,
        target_lengths
    )

    assert torch.isfinite(loss)

    print("Pooling + CTC integration test passed!")
    print("Pooled feature shape:", features.shape)
    print("CTC input lengths:", input_lengths.tolist())
    print("Loss:", loss.item())


if __name__ == "__main__":
    test_pooling_ctc()