

from parallel.utils import _next_power_of_2


def get_batches_train(seq_lens):
    n = seq_lens.size(0)
    batches = []
    r = n
    while (r > 0):
        take = 1_000_000
        batches.append((max(0, r - take), r))
        r -= take
    return batches

def get_batches_test(seq_lens):
    n = seq_lens.size(0)
    batches = []
    r = n
    VOLUME = int(2 ** 17) * 65
    while (r > 0):
        len = 1 + _next_power_of_2(seq_lens[r - 1].item() - 1)
        take = _next_power_of_2(max(1, VOLUME // len))
        batches.append((max(0, r - take), r))
        r -= take
    batches.sort()
    return batches