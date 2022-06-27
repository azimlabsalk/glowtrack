import glob
import os
import shutil
import sys


def deinterlace(image_dir, visible_dir, uv_dir):
    files = glob.glob(image_dir + '/*.*')

    os.makedirs(visible_dir, exist_ok=True)
    os.makedirs(uv_dir, exist_ok=True)

    for file in files:
        basename = os.path.basename(file)
        num_string = basename.split('.')[0]
        num = int(num_string)

        if num % 2 == 0:
            # uv
            num = num // 2
            d = uv_dir
        else:
            # visible
            num = (num - 1) // 2
            d = visible_dir

        new_file = '{:08d}.png'.format(num)
        new_file = os.path.join(d, new_file)
        shutil.move(file, new_file) 


if __name__ == '__main__':

    clip_dir = sys.argv[1]

    print(f'clip_dir: {clip_dir}')

    image_dir = clip_dir + '/interlaced.mp4-frames'
    uv_dir = clip_dir + '/uv'
    visible_dir = clip_dir + '/visible'

    deinterlace(image_dir, visible_dir, uv_dir)

