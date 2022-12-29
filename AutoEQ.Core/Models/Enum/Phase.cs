namespace AutoEQ.Core;

public enum Phase
{
    minimum,
    both,
    linear
}
/*
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

PEQ_CONFIGS = {
    "10_BAND_GRAPHIC_EQ': {
        'optimizer': {'min_std': 0.01},
        'filters': [{'fc': 31.25 * 2 ** i, 'q': math.sqrt(2), 'type': 'PEAKING'} for i in range(10)]
    },
    '10_PEAKING': {
        'filters': [{'type': 'PEAKING'}] * 10
    },
    '8_PEAKING_WITH_SHELVES': {
        'optimizer': {
            'min_std': 0.008
        },
        'filters': [{
            'type': 'LOW_SHELF',
            'fc': 105,
            'q': 0.7
        }, {
            'type': 'HIGH_SHELF',
            'fc': 10e3,
            'q': 0.7
        }] + [{'type': 'PEAKING'}] * 8
    },
    '4_PEAKING_WITH_LOW_SHELF': {
        'optimizer': {
            'max_f': 10000,
        },
        'filters': [{
            'type': 'LOW_SHELF',
            'fc': 105,
            'q': 0.7
        }] + [{'type': 'PEAKING'}] * 4
    },
    '4_PEAKING_WITH_HIGH_SHELF': {
        'filters': [{
            'type': 'HIGH_SHELF',
            'fc': 10000,
            'q': 0.7
        }] + [{'type': 'PEAKING'}] * 4
    },
}
*/