# -*- coding: utf-8 -*-

# Данные вида:
# 'ABC' - тип анализа
# ABC = [((R, G, B), значение), ...]
# ABC_units = 'единица измерения значений'
# ABC_corr = {значение : соотвествующая метка, ...}
# ABC_norm = значение для нормы анализа

# "-"       NEGATIVE
# "+/-"     TRACE
# "+"       LOW
# "++"      MODERATE
# "+++"     HIGH
# "++++"    VERY HIGH


# DIRUI
LEU_points = [
    (249, 246, 198),
    (240, 226, 190),
    (239, 210, 209),
    (166, 139, 164),
    (109, 73, 112)
]

LEU_target = [-1, 15, 70, 125, 500]

LEU_units = 'Leuko/uL'

LEU_corr = {
    -1: 'NEGATIVE',
    0: 'TRACE',
    1: 'LOW',
    2: 'MODERATE',
    3: 'HIGH'
}

LEU_norm = -1

# DIRUI
NIT_points = [
    (255, 253, 237),
    (253, 235, 228),
    (248, 203, 205),
]

NIT_target = [-1, 0, 1]

NIT_units = 'a.u.'

NIT_corr = {
    -1: 'NEGATIVE',
    0: 'HIGH',
    1: 'HIGH',
}

NIT_norm = -1

# DIRUI
URO_points = [
    (255, 252, 222),
    (249, 192, 169),
    (245, 167, 148),
    (243, 155, 131),
    (242, 143, 120)
]

URO_target = [3.4, 17, 34, 68, 135]

URO_units = '1 umol/L'

URO_corr = {
    3.4: 'NEGATIVE',
    17: 'TRACE',
    34: 'LOW',
    68: 'MODERATE',
    135: 'HIGH'
}

URO_norm = 17

# DIRUI
PRO_points = [
    (240, 233, 101),
    (182, 209, 105),
    (151, 199, 109),
    (141, 201, 163),
    (103, 192, 171),
    (13, 133, 145),
]

PRO_target = [-1, 0, 0.3, 1.0, 3.0, 20.0]

PRO_units = 'g/L'

PRO_corr = {
    -1: 'NEGATIVE',
    0: 'TRACE',
    0.3: 'LOW',
    1.0: 'MODERATE',
    3.0: 'LARGE',
    20.0: 'LARGE'
}

PRO_norm = 0

# DIRUI
PH_points = [
    (215, 212, 49),
    (168, 188, 54),
    (142, 175, 68),
    (117, 160, 52),
    (85, 151, 81),
    (72, 124, 85),
    (1, 102, 104),
]

PH_target = [5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]

PH_units = 'a.u.'

PH_corr = {
    5.0: '5.0',
    6.0: '6.0',
    6.5: '6.5',
    7.0: '7.0',
    7.5: '7.5',
    8.0: '8.0',
    8.5: '8.5',
}

PH_norm = 6.5

# DIRUI
BLO_points = [
    (251, 186, 33),
    (248, 172, 0), # + (86, 125, 52)
    (238, 180, 35),
    (213, 173, 19),
    (142, 141, 60), 
    (48, 120, 39),
]

BLO_units = 'Eryt/L' # ???

BLO_target = [-1, 9, 10, 25, 80, 200] # 9 и 10 - условно

BLO_corr = {
    -1: "NEGATIVE",
    9: "TRACE",
    10: "TRACE",
    25: "LOW",
    80: "MODERATE",
    200: "HIGH",
}

BLO_norm = 10

# DIRUI
SG_points = [
    (0, 73, 75),
    (114, 141, 68),
    (131, 149, 69),
    (146, 147, 73),
    (184, 156, 27),
    (213, 173, 19),
    (225, 187, 33)
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

SG_norm = 1.010

# DIRUI
KET_points = [
    (252, 217, 182),
    (246, 185, 172),
    (244, 165, 174),
    (211, 113, 145),
    (176, 65, 116),
    (122, 37, 84)
]

KET_target = [-1, 0.5, 1.5, 3.9, 7.8, 16]

KET_units = 'mmol/L'

KET_corr = {
    -1: 'NEGATIVE',
    0.5: 'TRACE',
    1.5: 'LOW',
    3.9: 'MODERATE',
    7.8: 'HIGH',
    16: 'VERY HIGH'
}

KET_norm = 0.5

# DIRUI
BIL_points = [
    (255, 250, 235),
    (253, 212, 150),
    (239, 177, 130),
    (226, 145, 118),
]

BIL_target = [-1, 17, 51, 103]

BIL_units = 'umol/L'

BIL_corr = {
    -1: 'NEGATIVE',
    1: 'LOW',
    2: 'MODERATE',
    3: 'HIGH'
}

BIL_norm = -1

# DIRUI
GLU_points = [
    (164, 215, 212),
    (150, 200, 120),
    (168, 188, 54),
    (181, 117, 60),
    (165, 89, 59),
]

GLU_target = [-1, 5.6, 14, 28, 56]

GLU_units = 'mmol/L'

GLU_corr = {
    -1: 'NEGATIVE',
    5.6: 'LOW   ',
    14: 'MODERATE',
    28: 'HIGH',
    56: 'VERY HIGH',
}

GLI_norm = -1

agents_list = ['LEU', 'NIT', 'URO', 'PRO', 'PH', 'BLO', 'SG', 'KET', 'BIL', 'GLU'][::-1]

points_dict = {
    'LEU': LEU_points,
    'NIT': NIT_points,
    'URO': URO_points,
    'PRO': PRO_points,
    'PH': PH_points,
    'BLO': BLO_points,
    'SG': SG_points,
    'KET': KET_points,
    'BIL': BIL_points,
    'GLU': GLU_points,
}

targets_dict = {
    'LEU': LEU_target,
    'NIT': NIT_target,
    'URO': URO_target,
    'PRO': PRO_target,
    'PH': PH_target,
    'BLO': BLO_target,
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
    'PH': PH_units,
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
    'PH': PH_corr,
    'BLO' : BLO_corr,
    'SG': SG_corr,
    'KET': KET_corr,
    'BIL': BIL_corr,
    'GLU': GLU_corr
}

norm_dict = {
    'LEU': LEU_norm,
    'NIT': NIT_norm,
    'URO': URO_norm,
    'PRO': PRO_norm,
    'PH': PH_norm,
    'BLO' : BLO_norm,
    'SG': SG_norm,
    'KET': KET_norm,
    'BIL': BIL_norm,
    'GLU': GLU_norm
}
