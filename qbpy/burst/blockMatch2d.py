import time
import numpy as np
from testing.TestFunctions import test_logger

@test_logger
def block_match_2d_multichannel(im0, im1, block_ul, block_size, search_radius, init_match=None, debug=False):
    """
    Brute-force block match in 2D for multi-channel images.

    Args:
        im0 (numpy array): Reference image.
        im1 (numpy array): Image to be matched.
        block_ul (list or tuple): Upper left corner coordinates of the block in im0 (0-based).
        block_size (int): Size of the block.
        search_radius (int): Radius of the search area.
        init_match (list or tuple, optional): Initial match coordinates (0-based). Default is [0, 0].
        debug (bool): If True, print debug information.

    Returns:
        tuple: (best_match, best_score)
            - best_match (array): Coordinates of the best match (0-based displacement from the block UL).
            - best_score (float): Score of the best match.
    """
    if init_match is None:
        init_match = [0, 0]

    h, w = im0.shape[0], im0.shape[1]
    c = im0.shape[2] if im0.ndim == 3 else 1

    # Upper left corner coordinates of the block in im0 (0-based)
    ylb, xlb = block_ul
    ylb -= 1 #  Convert Matlab indexing to python
    xlb -= 1

    # Extract the reference patch from im0 (multi-channel)
    if im0.ndim == 2:
        ref_patch = im0[ylb:ylb + block_size, xlb:xlb + block_size]
    else:
        ref_patch = im0[ylb:ylb + block_size, xlb:xlb + block_size, :]

    best_score = float('inf')
    best_match = [np.nan, np.nan]

    # Print debug info about reference patch
    if debug:
        print("[DEBUG] block_match_2d_multichannel:")
        print("  im0 shape:", im0.shape, "im1 shape:", im1.shape)
        print("  block_ul:", block_ul, "block_size:", block_size)
        print("  ref_patch shape:", ref_patch.shape)
        print("  search_radius:", search_radius, "init_match:", init_match)

    # Loop over search region
    v_min = int(max(-ylb, init_match[1] - search_radius))
    v_max = int(min(h - block_size - ylb, init_match[1] + search_radius))
    u_min = int(max(-xlb, init_match[0] - search_radius))
    u_max = int(min(w - block_size - xlb, init_match[0] + search_radius))

    if debug:
        print("  search ranges:")
        print("    v range:", v_min, "to", v_max)
        print("    u range:", u_min, "to", u_max)

    # move patch across reference image for template matching
    for v in range(v_min, v_max+1):
        t0_v = time.time()
        for u in range(u_min, u_max+1):
            # Extract current patch from im1 (multi-channel)
            if im1.ndim == 2:
                cur_patch = im1[ylb + v:ylb + v + block_size, xlb + u:xlb + u + block_size]
            else:
                cur_patch = im1[ylb + v:ylb + v + block_size, xlb + u:xlb + u + block_size, :]

            # Calculate the score as the sum of absolute differences across all channels
            cur_score = np.sum(np.abs(cur_patch - ref_patch))

            # Update the best match if the current score is better
            if cur_score < best_score:
                best_score = cur_score
                best_match = [u, v]

                if debug:
                    print("    New best score:", best_score, "at (u,v):", (u, v))
    if debug:
        print("  Final best_match:", best_match, "best_score:", best_score)

    return np.array([best_match]).squeeze(), best_score #wrap in list to match MATLAB output


