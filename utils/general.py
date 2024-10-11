def extract_layer_number(key):
    parts = key.split('.')
    for i, part in enumerate(parts):
        if part == 'layers':
            return int(parts[i + 1])
    return 0