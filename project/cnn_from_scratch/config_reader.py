def read_config(config_path="config.txt"):
    config = {}
    
    try:
        with open(config_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        if '.' in value:
                            config[key] = float(value)
                        else:
                            config[key] = int(value)
                    except ValueError:
                        config[key] = value
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using default values.")
        config = {
            'conv1_filters': 8,
            'conv1_kernel_size': 3,
            'conv1_input_channels': 1,
            'conv2_filters': 16,
            'conv2_kernel_size': 3,
            'pool_kernel_size': 2,
            'pool_stride': 2,
            'fc_output_size': 10,
            'input_height': 28,
            'input_width': 28
        }
    
    return config

def calculate_fc_input_size(config):
    height = config['input_height']
    width = config['input_width']
    
    height = height - config['conv1_kernel_size'] + 1
    width = width - config['conv1_kernel_size'] + 1
    height = height // config['pool_stride']
    width = width // config['pool_stride']
    
    height = height - config['conv2_kernel_size'] + 1
    width = width - config['conv2_kernel_size'] + 1
    height = height // config['pool_stride']
    width = width // config['pool_stride']
    
    return config['conv2_filters'] * height * width
