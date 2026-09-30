"""Field-plane projection helpers for calibrated, fixed-camera video."""
from __future__ import annotations

import cv2
import numpy as np


REEFSCAPE_FIELD_LENGTH_M = 17.548225  # AdvantageScope 690.875 in
REEFSCAPE_FIELD_WIDTH_M = 8.0518  # AdvantageScope 317 in

# Centers and vertices are traced from the bundled AdvantageScope field art.
# The dark central hex footprint is the carpet-plane obstacle; the larger
# colored hex is the Reef Zone boundary and must not be excluded.
_FIELD_CROP = (421.0, 91.0, 3352.0, 1437.0)
_ART_SCALE_X = 3773.0 / 2048.0
_ART_SCALE_Y = 1528.0 / 829.0
_REEF_ART_VERTICES = (
    ((634, 329), (709, 372), (709, 458), (634, 501), (559, 458), (559, 372)),
    ((1413, 329), (1488, 372), (1488, 458), (1413, 501), (1338, 458), (1338, 372)),
)


def reefscape_reef_footprints(field_length_m=REEFSCAPE_FIELD_LENGTH_M,
                              field_width_m=REEFSCAPE_FIELD_WIDTH_M):
    """Return the two 2025 Reef hardware footprints in wall-blue meters.

    Points are derived from the bundled AdvantageScope top-down field art and
    its documented field crop. They represent the physical hex bases only,
    not the larger Reef Zones.
    """
    left, top, right, bottom = _FIELD_CROP
    result = []
    for vertices in _REEF_ART_VERTICES:
        polygon = []
        for image_x, image_y in vertices:
            px, py = image_x * _ART_SCALE_X, image_y * _ART_SCALE_Y
            x = (right - px) / (right - left) * field_length_m
            y = (py - top) / (bottom - top) * field_width_m
            polygon.append((x, y))
        result.append(tuple(polygon))
    return tuple(result)


def point_in_polygon(point, polygon):
    """Return whether a 2D point lies inside or on a polygon boundary."""
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def reefscape_forbidden(point, field_length_m=REEFSCAPE_FIELD_LENGTH_M,
                        field_width_m=REEFSCAPE_FIELD_WIDTH_M):
    """True when a robot's projected floor point lies on a Reef base."""
    return any(point_in_polygon(point, polygon) for polygon in
               reefscape_reef_footprints(field_length_m, field_width_m))



def field_homography(image_corners, field_length_m=REEFSCAPE_FIELD_LENGTH_M,
                     field_width_m=REEFSCAPE_FIELD_WIDTH_M):
    """Build an image-to-field transform for floor-contact points."""
    corners = np.asarray(image_corners, dtype=np.float32).reshape(4, 2)
    if field_length_m <= 0 or field_width_m <= 0:
        raise ValueError("Field dimensions must be positive")
    if abs(cv2.contourArea(corners)) < 100:
        raise ValueError("Calibration corners must enclose a visible field area")
    dst = np.asarray([
        [field_length_m, 0], [0, 0], [0, field_width_m],
        [field_length_m, field_width_m],
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(corners, dst)
    if not np.isfinite(transform).all() or abs(np.linalg.det(transform)) < 1e-12:
        raise ValueError("Calibration corners do not define a valid perspective transform")
    return transform


def project_image_point(point, transform):
    """Project one image-space floor point through a field homography."""
    mapped = cv2.perspectiveTransform(
        np.asarray([[point]], dtype=np.float32), transform
    ).reshape(2)
    return float(mapped[0]), float(mapped[1])


def project_detection_rows(rows, image_corners, field_length_m=REEFSCAPE_FIELD_LENGTH_M,
                           field_width_m=REEFSCAPE_FIELD_WIDTH_M, exclude_reef=False):
    """Map ``[frame, x, y, confidence]`` floor points into field coordinates.

    ``image_corners`` map to the AdvantageScope Reefscape art in this order:
    red-top, blue-top, blue-bottom, red-bottom. Robot points should be
    bottom-center positions so the transform acts on the carpet plane.
    """
    # AdvantageScope's 2025 image has red on the left and blue on the right.
    # Wall-blue has its origin at the blue wall, +X toward red, and +Y toward
    # the bottom of the top-down image.
    transform = field_homography(image_corners, field_length_m, field_width_m)

    if not rows:
        return []
    image_points = np.asarray([[row[1], row[2]] for row in rows], dtype=np.float32).reshape(-1, 1, 2)
    field_points = cv2.perspectiveTransform(image_points, transform).reshape(-1, 2)
    projected = []
    for row, (x, y) in zip(rows, field_points):
        if (0 <= x <= field_length_m and 0 <= y <= field_width_m
                and not (exclude_reef and reefscape_forbidden((x, y), field_length_m, field_width_m))):
            projected.append([
                int(row[0]), round(float(x), 3), round(float(y), 3), round(float(row[3]), 3),
                *row[4:],
            ])
    return projected
