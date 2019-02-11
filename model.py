import torch
import numpy as np
from dataset import FashionMNIST
from linear import Conceptor
from nearest import Nearest_Neighbor
from perspective import *


class Active_Classifier:
    def __init__(self, device, num_classes, k=1, sample_size=(9, 9)):
        print("init")
        self.device = device
        self.num_classes = num_classes

        # width, height
        self.sample_size = sample_size
        self.perspective_dim = (1, 19, 19)
        self.perspective_count = self.perspective_dim[0] * self.perspective_dim[1] * self.perspective_dim[2]

        self.part = {}
        self.view_param = {}

        self.part[0] = get_perspective_kernels([[0, 0, 1], [0, 0, 1], [0, 0, 1]], scale=1)
        self.view_param[0] = get_perspective_kernels([[0, 0, self.perspective_dim[0]], [-0.2, 0.2, self.perspective_dim[1]], [-0.2, 0.2, self.perspective_dim[2]]], scale=1)

        self.part[1] = get_perspective_kernels([[0, 0, 1], [-0.5, 0.5, 3], [-0.5, 0.5, 3]], scale=2)
        self.view_param[1] = get_perspective_kernels([[0, 0, self.perspective_dim[0]], [-0.1, 0.1, self.perspective_dim[1]], [-0.1, 0.1, self.perspective_dim[2]]], scale=1)

        # self.part[2] = get_perspective_kernels([[0, 0, 1], [-0.2, 0.2, 3], [-0.2, 0.2, 3]], scale=2)
        # self.view_param[2] = get_perspective_kernels([[0, 0, self.perspective_dim[0]], [-0.05, 0.05, self.perspective_dim[1]], [-0.05, 0.05, self.perspective_dim[2]]], scale=1)

        self.models = {}
        self.episodic = [Nearest_Neighbor(device, output_is_tensor=False) for i in range(3)]
        self.semantic = Nearest_Neighbor(device, k=k)

        self.running_base_position = 0
        self.empty = True

    def to_tensor(self, input, dtype=torch.float32):
        return torch.tensor(input, dtype=dtype, device=self.device)

    def get_min_index(self, score):
        self.mid = self.perspective_count // 2
        score[:, self.mid] -= 1e-6
        index = torch.argmin(score, dim=1, keepdim=True)
        return index

    def sample_patches(self, input, layer, id="", base_perspective=None, force_center=False):
        batches = input.shape[0]

        patches = []
        ids = []
        best_scores = []

        if base_perspective is not None:
            part_perspective = np.reshape(rebase(self.part[layer], base_perspective), [-1, 2, 3])  # (batch*num part, 2, 3)
        else:
            part_perspective = np.tile(self.part[layer], (input.shape[0], 1, 1))

        count_parts = self.part[layer].shape[0]
        count_views = self.view_param[layer].shape[0]

        part_perspective = rebase(self.view_param[layer], part_perspective)  # (batch*num part, num perspective, 2, 3)
        perspectives = sample(input, self.to_tensor(np.reshape(part_perspective, [batches, -1, 2, 3])), size=self.sample_size)  # (batch, num part * num perspective, ...)
        perspectives = torch.reshape(perspectives, [batches, count_parts, count_views, -1])
        for i in range(count_parts):
            _flat = torch.reshape(perspectives[:, i, ...], [batches * count_views, -1])

            _id = id + str(i)
            if _id not in self.models:
                self.models[_id] = Conceptor(self.device, max_bases=20)

            if self.models[_id].get_count() == 0:
                _projected = torch.zeros(_flat.shape, device=self.device)
            else:
                _projected = self.models[_id].project(_flat)

            scores = torch.mean(torch.reshape((_flat - _projected)**2, [batches, count_views, -1]), dim=2)
            if force_center:
                min_index = torch.full([input.shape[0], 1], self.perspective_count // 2, device=self.device, dtype=torch.int64)
            else:
                min_index = self.get_min_index(scores)
            min_indices = torch.unsqueeze(min_index, 2).repeat(1, 1, perspectives.shape[3])
            min_perspective = torch.gather(perspectives[:, i, ...], 1, min_indices)[:, 0, ...]
            min_scores = torch.gather(scores, 1, min_index)[:, 0]

            patches.append(min_perspective)
            ids.append(_id)
            best_scores.append(min_scores)

            if layer < len(self.part) - 1:
                min_view_param = self.view_param[layer][np.squeeze(min_index.cpu().numpy()), ...]
                if len(min_view_param.shape) < 3:
                    min_view_param = np.expand_dims(min_view_param, 0)
                _ps, _ids, _bs = self.sample_patches(input, layer + 1, _id, min_view_param, force_center)
                patches += _ps
                ids += _ids
                best_scores += _bs

        return patches, ids, best_scores

    def reorder_bases(self, bases_list, id_list):
        ranks = []
        for _id in id_list:
            ranks.extend(self.models[_id].get_orders())
        _b = torch.cat(bases_list, dim=1)
        orders = np.argsort(np.array(ranks, dtype=np.int32))
        out = _b[:, orders]
        return out

    def forward(self, patches, ids, scores):
        batch = patches[0].shape[0]

        def __forward_loop_content(bj, index, id_list, bases_list):
            _id = ids[index]
            _patch = patches[index][bj:(bj + 1), ...]
            _bases = self.models[_id] << _patch
            bases_list.append(_bases)
            id_list.append(_id)
            return self.reorder_bases(bases_list, id_list)

        prediction_by_batch = []
        indices_by_batch = []
        for j in range(batch):

            id_list = []
            bases_list = []
            index_list = [0]
            for i in range(len(self.episodic)):
                bases = __forward_loop_content(j, index_list[i], id_list, bases_list)
                next_id = self.episodic[i] << bases
                index_list.append(ids.index(next_id[0][0]))
            bases = __forward_loop_content(j, index_list[-1], id_list, bases_list)

            prediction = (self.semantic << bases)
            prediction_by_batch.append(prediction)
            indices_by_batch.append(index_list)

        output_prediction = torch.cat(prediction_by_batch, dim=0)
        return output_prediction, indices_by_batch

    def backward(self, patches, ids, scores, indices, outputs):
        batch = patches[0].shape[0]

        def __backward_loop_content(bj, index, id_list, bases_list):
            _id = ids[index]
            _patch = patches[index][bj:(bj + 1), ...]
            self.running_base_position += self.models[_id].learn(_patch, 1, start_base_order=self.running_base_position, expand_threshold=1e-3)
            _bases = self.models[_id] << _patch
            bases_list.append(_bases)
            id_list.append(_id)
            return self.reorder_bases(bases_list, id_list)

        for j in range(batch):

            id_list = []
            bases_list = []
            for i in range(len(self.episodic)):
                bases = __backward_loop_content(j, indices[j][i], id_list, bases_list)
                self.episodic[i].learn(bases, [ids[indices[j][i + 1]]], num_classes=len(self.models))
            bases = __backward_loop_content(j, indices[j][len(self.episodic)], id_list, bases_list)

            self.semantic.learn(bases, outputs[j:(j + 1)], num_classes=self.num_classes)

    def classify(self, input, force_center=False):

        # patches = [tensor(batch, ...)]
        # ids = [id]
        # scores = [tensor(batch)]
        patches, ids, scores = self.sample_patches(input, 0, "", None, force_center)

        # prediction = tensor(batch)
        # res_ids_list = [[id1, id2, ...]]
        prediction, classify_indices_list = self.forward(patches, ids, scores)

        return prediction

    def classify_then_learn(self, input, output, force_center=False):

        batch = input.shape[0]

        # patches = [tensor(batch, ...)]
        # ids = [id]
        # scores = [tensor(batch)]
        patches, ids, scores = self.sample_patches(input, 0, "", None, force_center)

        _, salient_indices_list = torch.topk(torch.stack(scores, dim=1)[:, 1:], len(self.episodic), dim=1, largest=True)
        salient_indices_list = salient_indices_list + 1

        prediction = None
        classify_indices_list = None
        indices = [[0]] * batch

        if self.empty:
            for j in range(batch):
                for i in range(len(self.episodic)):
                    indices[j].append(salient_indices_list[j, i].item())
        else:
            # prediction = tensor(batch)
            # res_ids_list = [[id1, id2, ...]]
            prediction, classify_indices_list = self.forward(patches, ids, scores)
            correct_or_not = (prediction[:, 0] == output)

            for j in range(batch):
                for i in range(len(self.episodic)):
                    if correct_or_not[j]:
                        indices[j].append(classify_indices_list[j][i + 1])
                    else:
                        indices[j].append(salient_indices_list[j, i].item())

        self.backward(patches, ids, scores, indices, output)
        self.empty = False

        return prediction


if __name__ == '__main__':
    print("assert perspective.")

    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    display_size = (56, 56)
    classifier = Active_Classifier(device, 10, k=1, sample_size=display_size)

    batch_size = 10
    dataset = FashionMNIST(device, batch_size=batch_size, max_per_class=4, seed=0, group_size=1)

    for i, (data, label) in enumerate(dataset):
        print(data.shape, label.shape)
        print(dataset.readout(label))

        input = data.to(device)
        output = label.to(device)

        padded_input = np.pad(data.numpy()[:, 0, ...], [[0, 0], [0, display_size[1] - data.shape[2]], [0, display_size[0] - data.shape[3]]], 'constant')

        patches, _, __ = classifier.sample_patches(input, 0, "", None, False)
        classifier.classify_then_learn(input, output, True)

        gen = torch.reshape(torch.cat(patches, dim=0), [-1, batch_size, 1, display_size[0], display_size[1]])
        gen_cpu = gen.cpu().numpy()
        print(gen_cpu.shape)

        img = np.concatenate([
            1.0 - np.reshape(padded_input, [-1, display_size[0]]),
            np.reshape(np.transpose(gen_cpu[:, :, 0, ...], [1, 2, 0, 3]), [-1, display_size[0] * gen_cpu.shape[0]])
        ], axis=1)

        cv2.imshow("sample", img)
        cv2.waitKey(-1)
