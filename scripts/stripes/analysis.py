import numpy as np
import palette


def distance(p1, p2):
    """
    Calculate color difference between two points in RGB space.
    See https://www.compuphase.com/cmetric.htm.
    """

    assert len(p1) == len(p2) == 3, 'Points dimension mismatch: %s (dim p1) != %s (dim p2) != 3' % (len(p1), len(p2))

    r = (p1[0] + p2[0]) / 2.
    dr = p1[0] - p2[0]
    dg = p1[1] - p2[1]
    db = p1[2] - p2[2]

    d = (2 + r / 256.) * dr ** 2 + 4 * dg ** 2 + (2 + (255 - r) / 256.) * db ** 2

    return d


def find_nearest_idx(point, neighbors, metric=distance):
    """
    Find index of the closest neighbor in terms of distance metric.
    """
    min_dist = np.inf
    min_id = -1
    for j, neighbor in enumerate(neighbors):
        d = metric(point, neighbor)
        if d < min_dist:
            min_dist = d
            min_id = j
    return min_id


def analyze_one(image, pool_boxes, agent=None):
    """
    Take an image, array of pool_boxes and agent type, return a dictionary with analysis result.
    """

    if agent not in palette.agents_list:
        raise Exception("Agent not found")

    pool_idx = palette.agents_list.index(agent)
    points = palette.points_dict[agent]
    target = palette.targets_dict[agent]
    correspondence = palette.corr_dict[agent]
    units = palette.units_dict[agent]

    x1, y1, x2, y2 = pool_boxes[- pool_idx]
    pool = image[y1:y2, x1:x2, :]

    r = np.round(pool[:, :, 0].mean(), 2)
    g = np.round(pool[:, :, 1].mean(), 2)
    b = np.round(pool[:, :, 2].mean(), 2)

    nearest_idx = find_nearest_idx((r, g, b), points)

    result = {'agent': agent,
              'point': (r, g, b),
              'value': target[nearest_idx],
              'units': units,
              'summary': correspondence[target[nearest_idx]]}

    return result
