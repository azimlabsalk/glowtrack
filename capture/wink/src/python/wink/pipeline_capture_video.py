# Grab_MultipleCameras.cpp
# ============================================================================
# This sample illustrates how to grab and process images from multiple cameras
# using the CInstantCameraArray class. The CInstantCameraArray class represents
# an array of instant camera objects. It provides almost the same interface
# as the instant camera for grabbing.
# The main purpose of the CInstantCameraArray is to simplify waiting for images and
# camera events of multiple cameras in one thread. This is done by providing a single
# RetrieveResult method for all cameras in the array.
# Alternatively, the grabbing can be started using the internal grab loop threads
# of all cameras in the CInstantCameraArray. The grabbed images can then be processed by one or more
# image event handlers. Please note that this is not shown in this example.
# ============================================================================

from collections import namedtuple
import os
import sys
import time

#os.environ["PYLON_CAMEMU"] = "3"

import cv2
import click
import imageio
from pypylon import genicam
from pypylon import pylon

from wink.cameras import BaslerMultiCamera
from wink.pipeline import (Pairer, Indexer, MultiCameraStage, ThreadedWriterStage,
                           ProcessWriterStage)
from wink.triggers import ArduinoCameraTrigger
from wink.writers import ThreadedWriter, ProcessWriter


@click.command('multi_cam_record')
@click.argument('video-dir')
@click.option('--num-cams', type=int, default=2)
@click.option('--num-frames', type=int, default=1000)
@click.option('--quality', type=int, default=10)
@click.option('--writer-type', type=click.Choice(['thread', 'process']), default='thread')
@click.option('--deinterlace', type=bool, default=False)
def multi_cam_record(video_dir, num_cams, num_frames, quality, writer_type, deinterlace):

    print('video_dir: ', video_dir)
    print('num_cams: ', num_cams)
    print('num_frames: ', num_frames)
    print('quality: ', quality)
    print('writer_type: ', writer_type)
    print('deinterlace: ', deinterlace)

    os.makedirs(video_dir, exist_ok=True)

    if writer_type == 'thread':
        VideoWriterStage = ThreadedWriterStage
    elif writer_type == 'process':
        VideoWriterStage = ProcessWriterStage
    else:
        raise Exception("bad writer type:" + writer_type)

    def get_video_path(base_dir, camera_idx, light):
        camera_str = '{:02d}'.format(camera_idx)
        video_dir = os.path.join(base_dir, camera_str, light)
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, 'video.mp4')
        return video_path

    trigger = ArduinoCameraTrigger()
    trigger.stop_triggering()
    time.sleep(0.1)

    multi_cam = BaslerMultiCamera(num_cams=num_cams, trigger=trigger)
    multi_cam_stage = MultiCameraStage(multi_cam)

    writers = []
    camera_pipelines = []
    for i in range(num_cams):

        if deinterlace:

            pairer = Pairer()
            indexer0 = Indexer(0)
            indexer1 = Indexer(1)

            video_path0 = get_video_path(base_dir=video_dir,
                                         camera_idx=i,
                                         light='uv')

            video_path1 = get_video_path(base_dir=video_dir,
                                         camera_idx=i,
                                         light='visible')

            writer0 = VideoWriterStage(video_path0, quality=quality, fps=30,
                                       output_params=['-threads', '1'])
            writer1 = VideoWriterStage(video_path1, quality=quality, fps=30,
                                       output_params=['-threads', '1'])
            writers.append(writer0)
            writers.append(writer1)

            multi_cam_stage.connect(pairer)
            pairer.connect(indexer0)
            pairer.connect(indexer1)
            indexer0.connect(writer0)
            indexer1.connect(writer1)

        else:

            video_path = get_video_path(base_dir=video_dir,
                                        camera_idx=i,
                                        light='interlaced')

            writer = VideoWriterStage(video_path, quality=quality, fps=30,
                                      output_params=['-threads', '1'])
            writers.append(writer)

            multi_cam_stage.connect(writer)


    multi_cam_stage.capture(num_frames)

    while any([not writer.is_done() for writer in writers]):
        time.sleep(1)

    for writer in writers:
        writer.close()


if __name__ == '__main__':
    multi_cam_record()
