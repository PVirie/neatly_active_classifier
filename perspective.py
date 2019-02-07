import torch
import cv2
import numpy as np
import math
from dataset import FashionMNIST


def get_perspective_kernels(scope, scale=1):

    ts = np.linspace(scope[0][0], scope[0][1], scope[0][2])
    ys = np.linspace(scope[1][0] * scale, scope[1][1] * scale, scope[1][2])
    xs = np.linspace(scope[2][0] * scale, scope[2][1] * scale, scope[2][2])

    C = np.tile(np.reshape(np.cos(ts) / scale, [ts.shape[0], 1, 1]), [1, scope[1][2], scope[2][2]]).flatten()
    S = np.tile(np.reshape(np.sin(ts) / scale, [ts.shape[0], 1, 1]), [1, scope[1][2], scope[2][2]]).flatten()

    Y = np.tile(np.reshape(ys, [1, ys.shape[0], 1]), [scope[0][2], 1, scope[2][2]]).flatten()
    X = np.tile(np.reshape(xs, [1, 1, xs.shape[0]]), [scope[0][2], scope[1][2], 1]).flatten()

    Xt = C * X + S * Y
    Yt = -S * X + C * Y

    row0 = np.stack([C, S, -Xt], axis=1)
    row1 = np.stack([-S, C, -Yt], axis=1)

    thetas = np.stack([row0, row1], axis=1)
    return thetas


def rebase(affines, base):

    b = base.shape[0]
    p = affines.shape[0]

    A = np.reshape(affines, [1, p, 2, 3])
    B = np.reshape(base, [b, 1, 2, 3])

    C = A[:, :, 0, 0] * B[:, :, 0, 0] + A[:, :, 0, 1] * B[:, :, 1, 0]
    S = A[:, :, 1, 0] * B[:, :, 0, 0] + A[:, :, 1, 1] * B[:, :, 1, 0]

    Xt = A[:, :, 0, 0] * B[:, :, 0, 2] + A[:, :, 0, 1] * B[:, :, 1, 2] + A[:, :, 0, 2]
    Yt = A[:, :, 1, 0] * B[:, :, 0, 2] + A[:, :, 1, 1] * B[:, :, 1, 2] + A[:, :, 1, 2]

    row0 = np.stack([C, -S, Xt], axis=2)
    row1 = np.stack([S, C, Yt], axis=2)

    thetas = np.stack([row0, row1], axis=2)
    return thetas


def sample(x, affines, size=None):
    b = x.shape[0]
    p = affines.shape[1]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    out_h = size[1] if size is not None else h
    out_w = size[0] if size is not None else w

    thetas = torch.reshape(affines, [-1, 2, 3])
    cross = torch.reshape(x.unsqueeze(1).expand(-1, p, -1, -1, -1), [-1, c, h, w])

    grid = torch.nn.functional.affine_grid(thetas, [b * p, c, out_h, out_w])
    output = torch.reshape(torch.nn.functional.grid_sample(cross, grid), [b, p, c, out_h, out_w])
    return output


if __name__ == '__main__':
    print("assert perspective.")

    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 10
    dataset = FashionMNIST(device, batch_size=batch_size, max_per_class=2, seed=10, group_size=1)

    affines = np.tile(np.expand_dims(get_perspective_kernels([[0, math.pi / 6, 3], [0, 0.3, 5], [0, 0.3, 5]]), 0), [batch_size, 1, 1, 1])

    display_size = (56, 56)

    for j in range(4):

        for i, (data, label) in enumerate(dataset):
            print(data.shape, label.shape)
            print(dataset.readout(label))

            input = data.to(device)
            output = label.to(device)

            padded_input = np.pad(data.numpy()[:, 0, ...], [[0, 0], [0, display_size[1] - data.shape[2]], [0, display_size[0] - data.shape[3]]], 'constant')

            gen = sample(input, torch.tensor(affines, dtype=dtype, device=device), display_size)
            gen_cpu = gen.cpu().numpy()
            print(gen_cpu.shape)

            img = np.concatenate([
                1.0 - np.reshape(padded_input, [-1, display_size[0]]),
                np.reshape(np.transpose(gen_cpu[:, :, 0, ...], [0, 2, 1, 3]), [-1, display_size[0] * affines.shape[1]])
            ], axis=1)

            cv2.imshow("sample", img)
            cv2.waitKey(-1)

        print("Upscale...")

        base = affines[:, 0:1, ...]
        parts = get_perspective_kernels(
            [[0, 0, 1], [0, 1.0, 3], [0, 1.0, 3]],
            scale=2)
        affines = np.reshape(rebase(parts, base), [batch_size, -1, 2, 3])
