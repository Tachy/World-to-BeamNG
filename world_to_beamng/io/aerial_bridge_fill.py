"""
Retouches bridges out of the aerial photo.

The orthophoto shows the bridge deck from above, so the terrain below a bridge would carry a picture of the road (with
lane markings) draped over the valley floor. The real ground under the deck was never photographed; instead every pixel
of the bridge outline is filled with the texture beside the bridge, mirrored at the outline edge. Mirroring from both
sides and blending by distance keeps forest, rock and meadow texture sharp and lets features that cross under the
bridge (rivers, paths) continue through it - a plain inpainting blur would smear a 10-20 m wide strip.
"""

from typing import Iterable, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


def rasterize_areas(areas: Iterable[np.ndarray], bounds: Sequence[float], size: Tuple[int, int]) -> np.ndarray:
    """
    Boolean mask (rows, cols) of the photo pixels whose centers lie inside one of the `areas`.

    Args:
        areas: polygons as (N, 2) arrays in local coordinates
        bounds: (x_min, x_max, y_min, y_max) of the photo in local coordinates
        size: (width, height) of the photo in pixels - row 0 is the north edge
    """
    import shapely
    from shapely.geometry import Polygon

    x_min, x_max, y_min, y_max = bounds
    width, height = size
    px_x, px_y = (x_max - x_min) / width, (y_max - y_min) / height
    mask = np.zeros((height, width), dtype=bool)
    for area in areas:
        polygon = Polygon(np.asarray(area, dtype=float)[:, :2])
        if polygon.is_empty or not polygon.is_valid:
            polygon = polygon.buffer(0)
            if polygon.is_empty:
                continue
        a_min_x, a_min_y, a_max_x, a_max_y = polygon.bounds
        col0, col1 = max(0, int((a_min_x - x_min) / px_x)), min(width, int(np.ceil((a_max_x - x_min) / px_x)))
        row0, row1 = max(0, int((y_max - a_max_y) / px_y)), min(height, int(np.ceil((y_max - a_min_y) / px_y)))
        if col0 >= col1 or row0 >= row1:
            continue  # outside this photo
        cols, rows = np.meshgrid(np.arange(col0, col1), np.arange(row0, row1))
        inside = shapely.contains_xy(polygon, x_min + (cols + 0.5) * px_x, y_max - (rows + 0.5) * px_y)
        mask[row0:row1, col0:col1] |= inside
    return mask


def mirror_fill(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fills the masked pixels of `image` (rows, cols, channels) with the texture beside the mask.

    Each masked pixel looks along the direction away from its nearest outline edge: it takes the pixel mirrored at that
    near edge and the pixel mirrored at the opposite edge, weighted by closeness (the near side dominates at the
    edge, both sides meet halfway). A mirror source outside the image or inside the mask is dropped; if both are, the
    nearest unmasked pixel is used. Pixels outside the mask are returned unchanged.
    """
    mask = np.asarray(mask, dtype=bool)
    filled = image.copy()
    if not mask.any() or mask.all():
        return filled

    labels, _ = ndimage.label(mask)
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        # The farthest mirror source lies about one outline width beyond the edge: pad the crop by the short side
        pad = min(box[0].stop - box[0].start, box[1].stop - box[1].start) + 2
        crop = tuple(
            slice(max(0, s.start - pad), min(limit, s.stop + pad)) for s, limit in zip(box, mask.shape)
        )
        _fill_component(filled, image, mask[crop], labels[crop] == index, crop)
    return filled


def _fill_component(filled: np.ndarray, image: np.ndarray, sub_mask: np.ndarray, target: np.ndarray, crop) -> None:
    if sub_mask.all():
        return
    source = image[crop].astype(np.float32)
    distance, (near_rows, near_cols) = ndimage.distance_transform_edt(sub_mask, return_indices=True)

    rows, cols = np.nonzero(target)
    points = np.column_stack([rows, cols]).astype(np.float64)
    near = np.column_stack([near_rows[rows, cols], near_cols[rows, cols]]).astype(np.float64)
    near_distance = distance[rows, cols]
    direction = (points - near) / near_distance[:, None]  # points into the mask, away from the near edge

    far_distance = _distance_to_far_edge(sub_mask, points, direction)

    # Mirror at the pixel border between the edge pixel and the first masked pixel
    near_source = points - direction * (2.0 * near_distance - 1.0)[:, None]
    far_found = np.isfinite(far_distance)
    far_distance = np.where(far_found, far_distance, 0.0)
    far_source = points + direction * (2.0 * far_distance - 1.0)[:, None]
    near_ok, near_idx = _valid_sources(sub_mask, near_source)
    far_ok, far_idx = _valid_sources(sub_mask, far_source)
    far_ok &= far_found

    near_value = np.where(
        near_ok[:, None],
        source[near_idx[:, 0], near_idx[:, 1]],
        source[near[:, 0].astype(int), near[:, 1].astype(int)],
    )
    far_value = source[far_idx[:, 0], far_idx[:, 1]]
    near_weight = np.where(far_ok, far_distance / (near_distance + far_distance), 1.0)
    value = near_value * near_weight[:, None] + far_value * (1.0 - near_weight)[:, None]

    region = filled[crop]
    region[rows, cols] = np.clip(np.rint(value), 0, 255).astype(filled.dtype)


def _distance_to_far_edge(sub_mask: np.ndarray, points: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Steps from each point along its direction to the first unmasked pixel; inf if the ray leaves the crop first."""
    height, width = sub_mask.shape
    result = np.full(len(points), np.inf)
    active = np.arange(len(points))
    step = 0
    while active.size:
        step += 1
        probe = np.rint(points[active] + direction[active] * step).astype(int)
        outside = (probe[:, 0] < 0) | (probe[:, 0] >= height) | (probe[:, 1] < 0) | (probe[:, 1] >= width)
        inside = ~outside
        hit = np.zeros(active.size, dtype=bool)
        hit[inside] = ~sub_mask[probe[inside, 0], probe[inside, 1]]
        result[active[hit]] = step
        active = active[~hit & inside]
    return result


def _valid_sources(sub_mask: np.ndarray, sources: np.ndarray):
    """(valid, clipped integer index) of mirror sources: inside the crop and not masked."""
    height, width = sub_mask.shape
    index = np.rint(sources).astype(int)
    valid = (index[:, 0] >= 0) & (index[:, 0] < height) & (index[:, 1] >= 0) & (index[:, 1] < width)
    index = np.clip(index, 0, [height - 1, width - 1])
    valid &= ~sub_mask[index[:, 0], index[:, 1]]
    return valid, index


def fill_bridge_areas(photo: Image.Image, areas: Sequence[np.ndarray], bounds: Sequence[float]) -> Image.Image:
    """
    The aerial photo with the `areas` (bridge outlines in local coordinates) filled by mirror_fill(); the photo itself
    is returned when no area touches it.
    """
    if not len(areas):
        return photo
    mask = rasterize_areas(areas, bounds, photo.size)
    if not mask.any():
        return photo
    return Image.fromarray(mirror_fill(np.asarray(photo.convert("RGB")), mask))
