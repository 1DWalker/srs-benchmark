
def _next_power_of_2(value: int) -> int:
    if value < 1:
        raise ValueError("value must be at least 1.")
    return 1 << (value - 1).bit_length()