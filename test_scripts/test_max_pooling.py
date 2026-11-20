import torch
from src.model import MaxPoolingLayer


def test_max_pooling():
    x = torch.tensor([
        [[[1., 3., 2., 4.],
          [5., 6., 1., 2.],
          [2., 8., 3., 1.],
          [4., 2., 7., 3.]]]
    ])

    pooling = MaxPoolingLayer(kernel_size=2, stride=2)
    output = pooling(x)

    expected = torch.tensor([
        [[[6., 4.],
          [8., 7.]]]
    ])

    assert output.shape == (1, 1, 2, 2)
    assert torch.equal(output, expected)

    print("Max pooling test passed!")
    print("Output shape:", output.shape)
    print(output)


if __name__ == "__main__":
    test_max_pooling()