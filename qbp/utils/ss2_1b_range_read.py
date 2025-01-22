import os
from scipy.io import loadmat
import json

def ss2_1b_range_read(img_dir, start_frame, end_frame):
    # Read and parse the JSON file
    json_file_path = os.path.join(img_dir, 'info.json')
    with open(json_file_path, 'r') as f:
        seq_info = json.load(f)

    no_frames = seq_info['no_frames']
    start_part = (start_frame - 1) // no_frames + 1
    start_offset = start_frame - (start_part - 1) * no_frames
    end_part = (end_frame - 1) // no_frames + 1
    end_offset = end_frame - (end_part - 1) * no_frames

    imbs = [None] * (end_frame - start_frame)

    if start_part == end_part:
        # Single part
        part_path = os.path.join(img_dir, f'part_{start_part}.mat')
        data = loadmat(part_path)
        output = data['OUTPUT']
        for j in range(start_offset, end_offset + 1):
            imbs[j - start_offset] = output[:, :, j - 1]
    else:
        # Multiple parts
        # First part
        part_path = os.path.join(img_dir, f'part_{start_part}.mat')
        data = loadmat(part_path)
        output = data['OUTPUT']
        for j in range(start_offset, no_frames ):
            imbs[j - start_offset] = output[:, :, j]

        # Middle parts
        for i in range(start_part + 1, end_part):
            part_path = os.path.join(img_dir, f'part_{i}.mat')
            data = loadmat(part_path)
            output = data['OUTPUT']
            for j in range(no_frames):
                index = (i - start_part) * no_frames - start_offset + j
                imbs[index] = output[:, :, j]

        # Final part
        part_path = os.path.join(img_dir, f'part_{end_part}.mat')
        data = loadmat(part_path)
        output = data['OUTPUT']
        for j in range(end_offset):
            index = (end_part - start_part) * no_frames - start_offset + j
            imbs[index] = output[:, :, j]

    return imbs

# Example usage
# img_dir = "output_images"
# start_frame = 1
# end_frame = 100
# images = ss2_1b_range_read(img_dir, start_frame, end_frame)
