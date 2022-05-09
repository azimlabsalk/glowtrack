from collections import namedtuple
import time
import threading

import cv2
import numpy as np
from pypylon import pylon

frame_fields = ['image', 'camera_context', 'timestamp', 'frame_number', 'is_debayered']
Frame = namedtuple('Frame', frame_fields)

class BaslerCamera(object):

    def __init__(self, pylon_camera=None, exposure_time=1250.0):
        if pylon_camera is None:
            device = pylon.TlFactory.GetInstance().CreateFirstDevice()
            self.camera = pylon.InstantCamera(device)
        else:
            self.camera = pylon_camera

        self.model = self.camera.GetDeviceInfo().GetModelName()
        self.mono = (self.model[-1] == 'm')
        self.debayer = debayer
        self.exposure_time = exposure_time
        self.is_grabbing = False

    def start_grabbing(self, num_frames=None):
        self.is_grabbing = True
        self.camera.MaxNumBuffer = 15  # is this necessary?
        self.camera.Open()
        self.set_camera_params()

        if num_frames is not None:
            self.camera.StartGrabbingMax(num_frames,
                pylon.GrabStrategy_OneByOne,
                pylon.GrabLoop_ProvidedByInstantCamera)
        else:
            self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne,
                pylon.GrabLoop_ProvidedByInstantCamera)

    def stop_grabbing(self):
        self.is_grabbing = False
        self.camera.StopGrabbing()
        self.camera.Close()

    def set_camera_params(self):
        self.camera.CenterX.SetValue(True)
        self.camera.CenterY.SetValue(True)
        self.camera.Width.SetValue(848)
        self.camera.Height.SetValue(848)

        self.camera.TriggerMode.SetValue('On')
        self.camera.ExposureTime.SetValue(self.exposure_time)
        self.camera.Gain.SetValue(10.0)

        if not self.mono:
            # disable debayering (to save bandwidth by keeping image 1-channel)
            self.camera.PixelFormat.SetValue('BayerRG8')

    def register_handler(self, handler):
        self.camera_event_handler = CameraEventHandler(handler)
        self.camera.RegisterImageEventHandler(self.camera_event_handler,
            pylon.RegistrationMode_Append, pylon.Cleanup_Delete)


class CameraEventHandler(pylon.ImageEventHandler):

    def __init__(self, handler):
        super().__init__()
        self.handler = handler
#        self.debayer = debayer

    def OnImageGrabbed(self, camera, grab_result):

        if grab_result.GrabSucceeded():
#            camera = self.cameras[grab_result.GetCameraContext()]
#            cam_model = camera.GetDeviceInfo().GetModelName()
#            is_debayered = not is_color_cam(cam_model)
#            frame = make_frame(grab_result, is_debayered, debayer=self.debayer)
            self.handler(grab_result)
        else:
            print("Error: ", grab_result.GetErrorCode(),
                grab_result.GetErrorDescription())

        grab_result.Release()


class BaslerMultiCamera(object):

    def __init__(self, num_cams, trigger, debayer=True):
        self.num_cams = num_cams
        self.trigger = trigger
        self.cameras = None
        self.handler = None
        self.debayer = debayer

    def register_handler(self, handler):
        self.handler = handler

    def capture(self, num_frames):

        frames_to_grab = self.num_cams * num_frames

        self.setup_cameras()

        self.cameras.StartGrabbing()
        print('started grabbing')
        time.sleep(0.1)

        self.trigger.start_triggering()

        for i in range(frames_to_grab):

            if not self.cameras.IsGrabbing():
                break

            grab_result = self.cameras.RetrieveResult(5000,
                pylon.TimeoutHandling_ThrowException)

            if self.handler is not None:
                camera = self.cameras[grab_result.GetCameraContext()]
                cam_model = camera.GetDeviceInfo().GetModelName()
                is_debayered = not is_color_cam(cam_model)
                frame = make_frame(grab_result, is_debayered, debayer=self.debayer)
                self.handler(frame)

        self.trigger.stop_triggering()

    def setup_cameras(self):

        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()
        if len(devices) == 0:
            raise pylon.RUNTIME_EXCEPTION("No camera present.")

        self.num_cams = min(len(devices), self.num_cams)
        self.cameras = pylon.InstantCameraArray(self.num_cams)

        for i, camera in enumerate(self.cameras):
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


def is_color_cam(cam_model):
    return cam_model[-1] == 'c'


def make_frame(grab_result, is_debayered, debayer=False):

    image = grab_result.GetArray().copy()
    camera_context = grab_result.GetCameraContext()
    timestamp = grab_result.TimeStamp
    frame_number = grab_result.ImageNumber

    if debayer and not is_debayered:
        image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2BGR)
        is_debayered = True

    frame = Frame(image=image, frame_number=frame_number, timestamp=timestamp,
        camera_context=camera_context, is_debayered=is_debayered)

    return frame
