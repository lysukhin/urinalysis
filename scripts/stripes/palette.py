# -*- coding: utf-8 -*-

# Данные вида:
# 'ABC' - тип анализа
# ABC = [((R, G, B), значение), ...]
# ABC_units = 'единица измерения значений'
# ABC_corr = {значение : соотвествующая метка, ...}


LEU_points = [
    (229, 233, 182),
    (216, 212, 183),
    (188, 183, 163),
    (143, 117, 144),
    (129, 100, 144)
]

LEU_target = [-1, 0, 1, 2, 3]

LEU_units = 'a.u.'

LEU_corr = {
    -1: 'NEGATIVE',
    0: 'TRACE',
    1: 'SMALL',
    2: 'MODERATE',
    3: 'LARGE'
}

NIT_points = [
    (255, 255, 218),
    (255,239,206),
    (255,206,197),
]

NIT_units = 'a.u.'

NIT_corr = {
    -1: 'NEGATIVE',
    1: 'POSITIVE'
}

URO_points = [
    (254, 202, 152),
    (250, 168, 146),
    (235, 146, 142),
    (240, 138, 134),
    (232, 107, 137)
]

URO_target = [0.2, 1, 2, 4, 8]

URO_units = '1 mg/dL'

URO_corr = {
    0.2: 'NORMAL',
    1: 'NORMAL',
    2: 'ABOVE NORMAL',
    4: 'ABOVE NORMAL',
    8: 'ABOVE NORMAL'
}

PRO_points = [
    (220, 233, 119),
    (186, 211, 109),
    (165, 193, 119),
    (144, 185, 145),
    (112, 175, 154),
    (89, 156, 138),
]

PRO_target = [-1, 0, 1, 2, 3, 4]

PRO_units = 'a.u.'

PRO_corr = {
    -1: 'NEGATIVE',
    0: 'TRACE',
    1: '30 mg/ml',
    2: '100 mg/ml',
    3: '300 mg/ml',
    4: '2000+ mg/ml'
}

pH_points = [
    (241, 129, 74),
    (236, 164, 82),
    (220, 192, 83),
    (181, 193, 91),
    (129, 165, 77),
    (76, 155, 107),
    (28, 122, 124),
]

pH_target = [5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]

pH_units = 'a.u.'

pH_corr = {
    5.0: '5.0',
    6.0: '6.0',
    6.5: '6.5',
    7.0: '7.0',
    7.5: '7.5',
    8.0: '8.0',
    8.5: '8.5',
}

BLO_points = [
    (229, 190, 63),
    (230,189,56),
    (219,201,70),
    (193,191,68),
    (146,171,69),
    (87,136,70),
    (57,83,58), 
    # TODO
]

BLO_units = 'a.u.'

# TODO
# BLO_corr =


SG_points = [
    (17, 73, 72),
    (38, 113, 72),
    (95, 124, 180),
    (119, 136, 58),
    (140, 147, 54),
    (139, 143, 48),
    (184, 160, 55)
]

SG_target = [1.000, 1.005, 1.010, 1.015, 1.020, 1.025, 1.030]

SG_units = 'a.u.'

SG_corr = {
    1.000: '1.000',
    1.005: '1.005',
    1.010: '1.010',
    1.015: '1.015',
    1.020: '1.020',
    1.025: '1.025',
    1.030: '1.030',
}

KET_points = [
    (215, 182, 151),
    (232, 175, 156),
    (214, 128, 127),
    (178, 97, 103),
    (134, 72, 85),
    (97, 54, 64)
]

KET_target = [-1, 5, 15, 40, 80, 160]

KET_units = 'mg/dL'

KET_corr = {
    -1: 'NEGATIVE',
    5: 'TRACE',
    15: 'SMALL',
    40: 'MODERATE',
    80: 'LARGE',
    160: 'LARGE'
}

BIL_points = [
    (264, 231, 164),
    (237, 216, 164),
    (200, 195, 155),
    (191, 173, 149),
]

BIL_target = [-1, 1, 2, 3]

BIL_units = 'a.u.'

BIL_corr = {
    -1: 'NEGATIVE',
    1: 'SMALL',
    2: 'MODERATE',
    3: 'LARGE'
}

GLU = [
    (144, 208, 182),
    (140, 196, 135),
    (123, 176, 86),
    (140, 144, 67),
    (129, 118, 54),
    (119, 85, 58),
]

GLU_target = [-1, 100, 250, 500, 1000, 2000]

GLU_units = 'mg/dL'

GLU_corr = {
    -1: 'NEGATIVE',
    100: '100',
    250: '250',
    500: '500',
    1000: '1000',
    2000: '2000'
}

agents_list = ['LEU', 'NIT', 'URO', 'PRO', 'pH', 'BLO', 'SG', 'KET', 'BIL', 'GLU'][::-1]

points_dict = {
    'LEU': LEU_points,
     'NIT': NIT_points,
    'URO': URO_points,
    'PRO': PRO_points,
    'pH': pH_points,
     'BLO': BLO_points,
    'SG': SG_points,
    'KET': KET_points,
    'BIL': BIL_points,
    'GLU': GLU
}

targets_dict = {
    'LEU': LEU_target,
     #'NIT': NIT_target,
    'URO': URO_target,
    'PRO': PRO_target,
    'pH': pH_target,
     #'BLO': BLO_target,
    'SG': SG_target,
    'KET': KET_target,
    'BIL': BIL_target,
    'GLU': GLU_target
}

units_dict = {
    'LEU': LEU_units,
    'NIT': NIT_units,
    'URO': URO_units,
    'PRO': PRO_units,
    'pH': pH_units,
    'BLO': BLO_units,
    'SG': SG_units,
    'KET': KET_units,
    'BIL': BIL_units,
    'GLU': GLU_units
}

corr_dict = {
    'LEU': LEU_corr,
    'NIT': NIT_corr,
    'URO': URO_corr,
    'PRO': PRO_corr,
    'pH': pH_corr,
    # 'BLO' : BLO_corr,
    'SG': SG_corr,
    'KET': KET_corr,
    'BIL': BIL_corr,
    'GLU': GLU_corr
}
