

from parallel.utils import _next_power_of_2


def get_batches_train(seq_lens):
    n = seq_lens.size(0)
    batches = []
    r = n
    SIZE = int(2 ** 15)
    while (r > 0):
        batches.append((max(0, r - SIZE), r))
        r -= SIZE
    return batches

def get_batches_test(seq_lens):
    n = seq_lens.size(0)
    batches = []
    r = n
    VOLUME = int(2 ** 16) * 64
    while (r > 0):
        len = _next_power_of_2(seq_lens[r - 1].item())
        take = max(1, VOLUME // len)
        batches.append((max(0, r - take), r))
        r -= take
    batches.sort()
    return batches