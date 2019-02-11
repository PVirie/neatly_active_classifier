import torch
import numpy as np
import cv2
from dataset import FashionMNIST
from model import Active_Classifier


if __name__ == "__main__":
    print("main")

    device = torch.device("cuda:0")

    batch_size = 1
    dataset = FashionMNIST(device, batch_size=batch_size, max_per_class=60, seed=0, group_size=2)

    classifier = Active_Classifier(device, 10, k=1)

    percent_correct = 0.0
    for i, (data, label) in enumerate(dataset):
        print("data: ", i)

        input = data.to(device)
        output = label.to(device)

        # online test
        prediction = classifier.classify_then_learn(input, output, i < 20)

        if prediction is not None:
            prediction_cpu = prediction.cpu()
            correct = (prediction[:, 0] == output)
            count_correct = np.sum(correct.cpu().numpy())
            percent_correct = 0.99 * percent_correct + 0.01 * count_correct * 100 / batch_size
            print("Truth: ", dataset.readout(label))
            print("Guess: ", dataset.readout(prediction_cpu.flatten()))
            print("Percent correct: ", percent_correct)

        img = np.reshape(data.numpy(), [-1, data.shape[2]])
        cv2.imshow("sample", img)
        cv2.waitKey(10)

    print("Computing backward scores...")
    count = 0
    for i, (data, label) in enumerate(dataset):
        input = data.to(device)
        output = label.to(device)

        # test
        prediction = classifier.classify(input).cpu()
        count = count + np.sum(prediction.numpy()[:, 0] == label.numpy())

    print("Percent correct: ", count * 100 / (len(dataset) * batch_size))
