"""
wave function collapse (WFC) module
-----------------------------------
Lightweight, self-contained Wave Function Collapse (WFC) for tiled images.

API (main bits):
- wave_function_collapse(input_image: np.ndarray, pattern_size: int, out_h: int, out_w: int, flip: bool, rotate: bool) -> np.ndarray
- collapse_from_patterns(patterns: np.ndarray, frequencies: list[int], out_h: int, out_w: int) -> np.ndarray

Patterns are N x N x C tiles (uint8). Output is an H x W x C mosaic stitched from the top-left pixel of
the chosen tile per cell (so use pattern_size=1 for working on code-index grids, or any N for RGB tiles).
"""
import queue
from typing import Dict, List, Set, Tuple, Any, Union

import numpy as np

# Status constants
CONTRADICTION = -2
WAVE_COLLAPSED = -1
RUNNING = 0

def to_tuple(arr: np.ndarray) -> Union[Tuple, np.ndarray]:
    if isinstance(arr, np.ndarray):
        return tuple(map(to_tuple, arr))
    return arr

def to_ndarray(tup: Tuple) -> Union[np.ndarray, Tuple]:
    if isinstance(tup, tuple):
        return np.array(list(map(to_ndarray, tup)))
    return tup

def get_dirs(n: int) -> List[Tuple[int, int]]:
    dirs = [(i, j) for j in range(-n + 1, n) for i in range(-n + 1, n)]
    dirs.remove((0, 0))
    return dirs

def flip_dir(d: Tuple[int, int]) -> Tuple[int, int]:
    return (-d[0], -d[1])

def mask_with_offset(pattern: np.ndarray, offset: Tuple[int, int]) -> np.ndarray:
    x_off, y_off = offset
    if abs(x_off) > pattern.shape[0] or abs(y_off) > pattern.shape[1]:
        return np.array([[]])
    return pattern[max(0, x_off):min(pattern.shape[0], pattern.shape[0] + x_off),
                   max(0, y_off):min(pattern.shape[1], pattern.shape[1] + y_off),
                   :]

def check_for_match(p1: np.ndarray, p2: np.ndarray, offset: Tuple[int, int]) -> bool:
    p1_off = mask_with_offset(p1, offset)
    p2_off = mask_with_offset(p2, flip_dir(offset))
    return np.array_equal(p1_off, p2_off)

def get_rules(patterns: np.ndarray, directions: List[Tuple[int, int]]) -> List[Dict[Tuple[int, int], Set[int]]]:
    rules = [{d: set() for d in directions} for _ in range(len(patterns))]
    for i in range(len(patterns)):
        for d in directions:
            for j in range(i, len(patterns)):
                if check_for_match(patterns[i], patterns[j], d):
                    rules[i][d].add(j)
                    rules[j][flip_dir(d)].add(i)
    return rules

def extract_patterns_and_freq(im: np.ndarray, N: int, flip: bool = True, rotate: bool = True) -> Tuple[np.ndarray, List[int]]:
    H, W, _ = im.shape
    tiles = []
    for i in range(H - N + 1):
        for j in range(W - N + 1):
            tiles.append(im[i:i+N, j:j+N, :])
    if flip:
        tiles += [np.flip(t, axis=0) for t in tiles]
        tiles += [np.flip(t, axis=1) for t in tiles]
    if rotate:
        tiles += [np.rot90(t, k=1) for t in tiles]
        tiles += [np.rot90(t, k=2) for t in tiles]
        tiles += [np.rot90(t, k=3) for t in tiles]
    # count unique tiles by tuple key
    keys = [to_tuple(t) for t in tiles]
    uniq = {}
    for k in keys:
        uniq[k] = uniq.get(k, 0) + 1
    patterns = np.array([to_ndarray(k) for k in uniq.keys()])
    freqs = list(uniq.values())
    return patterns, freqs

def initialize(patterns: np.ndarray, freqs: List[int], out_h: int, out_w: int, pattern_size: int):
    directions = get_dirs(pattern_size)
    rules = get_rules(patterns, directions)
    coeff = np.full((out_h, out_w, len(patterns)), True, dtype=bool)
    return coeff, directions, freqs, patterns, rules

def is_collapsed(coeff: np.ndarray, pos: Tuple[int, int]) -> bool:
    return np.sum(coeff[pos[0], pos[1], :]) == 1

def in_bounds(pos: Tuple[int, int], shape: Tuple[int, ...]) -> bool:
    return 0 <= pos[0] < shape[0] and 0 <= pos[1] < shape[1]

def min_entropy_cell(coeff: np.ndarray, freqs: List[int]) -> Tuple[int, int, int]:
    prob = np.array(freqs, dtype=float)
    prob = prob / prob.sum()
    ent = np.sum(coeff.astype(int) * prob, axis=2)
    ent[np.sum(coeff, axis=2) == 1] = 0
    if np.sum(ent) == 0:
        return -1, -1, WAVE_COLLAPSED
    min_val = np.min(ent[np.where(ent > 0)])
    ys, xs = np.where(ent == min_val)
    idx = np.random.randint(len(xs))
    return ys[idx], xs[idx], RUNNING

def collapse_cell(coeff: np.ndarray, freqs: List[int], pos: Tuple[int, int]) -> np.ndarray:
    options = np.where(coeff[pos[0], pos[1]])[0]
    weights = np.array(freqs)[options]
    pick = np.random.choice(options, p=weights/weights.sum())
    cell = np.zeros_like(coeff[pos[0], pos[1]])
    cell[pick] = True
    coeff[pos[0], pos[1]] = cell
    return coeff

def propagate_from(pos: Tuple[int, int], coeff: np.ndarray, rules, directions) -> np.ndarray:
    q = queue.Queue()
    q.put(pos)
    while not q.empty():
        r, c = q.get()
        for d in directions:
            rr, cc = r + d[0], c + d[1]
            if not in_bounds((rr, cc), coeff.shape): 
                continue
            if is_collapsed(coeff, (rr, cc)):
                continue
            source_opts = np.where(coeff[r, c])[0]
            possible = np.zeros(coeff.shape[2], dtype=bool)
            for s in source_opts:
                for t in rules[s][d]:
                    possible[t] = True
            new_cell = np.logical_and(possible, coeff[rr, cc])
            if not np.array_equal(new_cell, coeff[rr, cc]):
                coeff[rr, cc] = new_cell
                q.put((rr, cc))
    return coeff

def observe(coeff: np.ndarray, freqs: List[int]) -> Tuple[Tuple[int, int], np.ndarray, int]:
    if np.any(~np.any(coeff, axis=2)):
        return (-1, -1), coeff, CONTRADICTION
    y, x, state = min_entropy_cell(coeff, freqs)
    if state == WAVE_COLLAPSED:
        return (y, x), coeff, WAVE_COLLAPSED
    coeff = collapse_cell(coeff, freqs, (y, x))
    return (y, x), coeff, RUNNING

def image_from_coeffs(coeff: np.ndarray, patterns: np.ndarray) -> Tuple[int, np.ndarray]:
    H, W, _ = coeff.shape
    # simple visualization: mean over valid patterns' top-left pixel
    out = np.zeros((H, W, patterns.shape[-1]), dtype=float)
    collapsed = 0
    for i in range(H):
        for j in range(W):
            ids = np.where(coeff[i, j])[0]
            pix = patterns[ids, 0, 0, :].mean(axis=0) if len(ids) > 0 else 0
            out[i, j] = pix
            if len(ids) == 1:
                collapsed += 1
    return collapsed, out.astype(np.uint8)

def collapse_from_patterns(patterns: np.ndarray, freqs: List[int], out_h: int, out_w: int) -> np.ndarray:
    N = patterns.shape[1]
    coeff, directions, freqs, patterns, rules = initialize(patterns, freqs, out_h, out_w, N)
    state = RUNNING
    while state != WAVE_COLLAPSED:
        pos, coeff, state = observe(coeff, freqs)
        if state == CONTRADICTION:
            raise RuntimeError("WFC contradiction occurred (no valid patterns left).")
        coeff = propagate_from(pos, coeff, rules, directions)
    _, img = image_from_coeffs(coeff, patterns)
    return img

def wave_function_collapse(input_image: np.ndarray, pattern_size: int, out_h: int, out_w: int,
                           flip: bool = False, rotate: bool = False) -> np.ndarray:
    patterns, freqs = extract_patterns_and_freq(input_image, pattern_size, flip=flip, rotate=rotate)
    return collapse_from_patterns(patterns, freqs, out_h, out_w)
