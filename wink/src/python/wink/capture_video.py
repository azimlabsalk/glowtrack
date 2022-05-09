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

import os
import sys
import time


#os.environ["PYLON_CAMEMU"] = "3"

import cv2
import click
import imageio
from pypylon import genicam
from pypylon import pylon

from wink.triggers import ArduinoCameraTrigger
from wink.writers import ThreadedWriter, ProcessWriter


@click.command('multi_cam_record')
@click.argument('video-dir')
@click.option('--num-cams', type=int, default=2)
@click.option('--num-frames', type=int, default=1000)
@click.option('--quality', type=int, default=10)
@click.option('--writer-type', type=click.Choice(['thread', 'process']), default='thread')
@click.option('--deinterlace', default=False)
def multi_cam_record(video_dir, num_cams, num_frames, quality, writer_type, deinterlace):

    print('video_dir: ', video_dir)
    print('num_cams: ', num_cams)
    print('num_frames: ', num_frames)
    print('quality: ', quality)
    print('writer_type: ', writer_type)
    print('deinterlace: ', deinterlace)

    if writer_type == 'thread':
        WriterClass = ThreadedWriter
    elif writer_type == 'process':
        WriterClass = ProcessWriter
    else:
        raise Exception("bad writer type:" + writer_type)

    os.makedirs(video_dir, exist_ok=True)

    countOfImagesToGrab = num_cams * num_frames
    maxCamerasToUse = num_cams

    # The exit code of the sample application.
    exitCode = 0

    trigger = ArduinoCameraTrigger()
    time.sleep(0.1)

    writers = {}
    timestamp_files = []
    for i in range(maxCamerasToUse):
        timestamp_path = os.path.join(video_dir, '{:02d}.txt'.format(i))
        timestamp_files.append(timestamp_path)
        writer_path = os.path.join(video_dir, '{:02d}.mp4'.format(i))
        writers[i] = WriterClass(writer_path, quality=quality, fps=30, output_params=['-threads', '1'])

    try:

        # Get the transport layer factory.
        tlFactory = pylon.TlFactory.GetInstance()

        # Get all attached devices and exit application if no device is found.
        devices = tlFactory.EnumerateDevices()
        if len(devices) == 0:
            raise pylon.RUNTIME_EXCEPTION("No camera present.")

        # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
        cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))

        l = cameras.GetSize()
        print('cameras.GetSize() = {}'.format(l))

        # Create and attach all Pylon Devices.
        for i, camera in enumerate(cameras):
            camera.Attach(tlFactory.CreateDevice(devices[i]))

            # Print the model name of the camera.
            print("Using device ", camera.GetDeviceInfo().GetModelName())

            camera.MaxNumBuffer = 15

            camera.Open()

            camera.CenterX.SetValue(True)
            camera.CenterY.SetValue(True)
            camera.Width.SetValue(848)
            camera.Height.SetValue(848)

            camera.TriggerMode.SetValue('On')
            camera.ExposureTime.SetValue(1250.0)
            camera.Gain.SetValue(10.0)

            try:
                camera.PixelFormat.SetValue('BayerRG8')  # faster (but need debayering)
            except Exception as e:
                print(e)

        # Starts grabbing for all cameras starting with index 0. The grabbing
        # is started for one camera after the other. That's why the images of all
        # cameras are not taken at the same time.
        # However, a hardware trigger setup can be used to cause all cameras to grab images synchronously.
        # According to their default configuration, the cameras are
        # set up for free-running continuous acquisition.
        trigger.stop_triggering()
        time.sleep(0.1)

        cameras.StartGrabbing()
        print('started grabbing')
        time.sleep(0.1)

        trigger.start_triggering()

        # Grab c_countOfImagesToGrab from the cameras.
        for i in range(countOfImagesToGrab):
            if not cameras.IsGrabbing():
                break

            grabResult = cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            # When the cameras in the array are created the camera context value
            # is set to the index of the camera in the array.
            # The camera context is a user settable value.
            # This value is attached to each grab result and can be used
            # to determine the camera that produced the grab result.
            cameraContextValue = grabResult.GetCameraContext()
            # import pdb; pdb.set_trace()

            # Print the index and the model name of the camera.
            cam_model = cameras[cameraContextValue].GetDeviceInfo().GetModelName()
            print("Camera ", cameraContextValue, ": ", cam_model)

            # Now, the image data can be processed.
            # print("GrabSucceeded: ", grabResult.GrabSucceeded())
            print("GetImageNumber: ", grabResult.GetImageNumber())
            print("TimeStamp: ", grabResult.TimeStamp)
            img = grabResult.GetArray()
            print("Gray value of first pixel: ", img[0, 0])
            print("SkippedImages: " + str(grabResult.GetNumberOfSkippedImages()))

            if cam_model[-1] == 'c':
                img = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2BGR)

            writer = writers[cameraContextValue]
            writer.append_data(img)

            with open(timestamp_files[cameraContextValue], 'a') as f:
                f.write(str(grabResult.TimeStamp) + '\n')

    except genicam.GenericException as e:
        # Error handling
        print("An exception occurred.", e)
        exitCode = 1

    trigger.stop_triggering()

    while any([not writer.is_done() for writer in writers.values()]):
        time.sleep(1)

    for writer in writers.values():
        writer.close()

    return exitCode


# @click.command('multi_cam_record')
# @click.argument('video-dir', default='.')
# @click.option('--num-cams', type=int, default=2)
# @click.option('--num-frames', type=int, default=1000)
# @click.option('--quality', type=int, default=10)
# @click.option('--writer-type', type=click.Choice(['thread', 'process']), default='thread')
# def multi_cam_record(video_dir, num_cams, num_frames, quality, writer_type):
#     video_dir = os.path.realpath(video_dir)
#
#     print('video_dir: ', video_dir)
#     print('num_cams: ', num_cams)
#     print('num_frames: ', num_frames)
#     print('quality: ', quality)
#     print('writer_type: ', writer_type)
#
#     multicam = MultiCamera(video_dir=video_dir, num_cams=num_cams,
#                            quality=quality)
#
#     print('multicam.record()')
#     multicam.record(num_frames, writer_type)
#
#     print('multicam.write_metadata()')
#     multicam.write_metadata(dirpath='metadata')
#
#     for i in range(multicam.num_cams):
#         desired_time = 5000000
#         time_errors = np.diff(multicam.timestamps[i]) - desired_time
#         late_frames = np.sum(np.abs(time_errors) > 100000)
#         print('Cam {} had {} late frames'.format(i, late_frames))


if __name__ == '__main__':
    exit_code = multi_cam_record()
    sys.exit(exit_code)
