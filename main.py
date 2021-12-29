"""Creates Voronoi diagram."""
import numpy as np
import math
from matplotlib import pyplot as plt

pixels = np.zeros((512, 512, 3), dtype=np.uint8)
pixels[255, 255] = [255, 0, 0]

centres = list(map(lambda x: np.random.choice(range(512), size=2), range(10)))

colours = list(map(lambda x: np.random.choice(range(256), size=3), centres))


def euclidean_distance(point_a, point_b):
    """Compute the Euclidean distance between two points."""
    return math.sqrt(
        (point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2
    )


def find_nearest_centre(point):
    """Find the nearest centre to a point."""
    min_index = 0
    min_distance = euclidean_distance(point, centres[0])

    for i in range(1, len(centres)):
        distance = euclidean_distance(point, centres[i])

        if distance < min_distance:
            min_index = i
            min_distance = distance

    return min_index


for row in range(512):
    for column in range(512):
        centre_index = find_nearest_centre((row, column))
        pixels[row, column] = colours[centre_index]

for centre in centres:
    pixels[centre[0], centre[1]] = [255, 0, 0]

plt.imshow(pixels, interpolation='nearest')
plt.show()
