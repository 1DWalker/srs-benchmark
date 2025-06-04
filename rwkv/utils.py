from io import BytesIO
import torch
from collections import deque


def get_number_of_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def transfer_child_grad_to_master(master, child):
    master_params = dict(master.named_parameters())
    for name, param in child.named_parameters():
        # print(name, param.grad)
        master_param = master_params[name]
        if (
            param.grad is not None
        ):  # None happens on the first few iterations for some params
            # Add the child model's grad
            with torch.no_grad():
                if master_param.grad is None:
                    master_param.grad = torch.zeros_like(
                        master_param, requires_grad=True
                    )
                master_param.grad.add_(param.grad.to(torch.float32))
            # Set the child model's grad to zero
            param.grad.zero_()

class SlidingWindowAverage:
    def __init__(self, len: int):
        self.len = len
        self.queue = deque()
        self.tot = 0
        self.n = 0

    def at_capacity(self):
        return self.len == len(self.queue)

    def add_value(self, avg, weight=1):
        self.tot += avg * weight
        self.n += weight

        self.queue.append((avg, weight))
        if len(self.queue) > self.len:
            prev_avg, prev_n = self.queue.popleft()
            self.tot -= prev_avg * prev_n
            self.n -= prev_n

    def get_value(self):
        assert self.n > 0
        return self.tot / self.n


class KeyValueAverage:
    def __init__(self):
        self.values = {}
        self.weights = {}
        self.tot = 0
        self.n = 0

    def add_value(self, key, avg, weight=1):
        if key not in self.values:
            self.values[key] = 0
            self.weights[key] = 0

        self.tot -= self.values[key] * self.weights[key]
        self.n -= self.weights[key]
        self.values[key] = avg
        self.weights[key] = weight
        self.tot += self.values[key] * self.weights[key]
        self.n += self.weights[key]

    def get_value(self):
        assert self.n > 0
        return self.tot / self.n
