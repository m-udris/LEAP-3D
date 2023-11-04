def normalize_temperature_2d(x, melting_point, base_temperature=0, inplace=False, inverse=False):
    if not inplace:
        x = x.clone()

    if len(x.shape) == 2:
        x = _normalize_temperature(x, melting_point, base_temperature, inverse=inverse)
        return x
    x[:, :, -1] = _normalize_temperature(x[:, :, -1], melting_point, base_temperature, inverse=inverse)
    return x

def normalize_temperature_2d(x, melting_point, base_temperature=0, inplace=False, inverse=False):
    if not inplace:
        x = x.clone()

    if len(x.shape) == 2:
        x = _normalize_temperature(x, melting_point, base_temperature, inverse=inverse)
        return x
    x[:, :, -1] = _normalize_temperature(x[:, :, -1], melting_point, base_temperature, inverse=inverse)
    return x


def _normalize_temperature(x, melting_point, base_temperature, inverse=False):
    if inverse:
        return x * (melting_point - base_temperature) + base_temperature
    return (x - base_temperature) / (melting_point - base_temperature)