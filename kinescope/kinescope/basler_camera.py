from collections import namedtuple
import time
import threading

import cv2
import numpy as np
from pypylon import pylon
from skimage.io import imsave

from realtime_pipeline import Stage

Frame = namedtuple('Frame', ['timestamp', 'image'])

class BaslerCamera(Stage):

    def __init__(self, pylon_camera=None, exposure_time=1250.0, trigger_mode=True):
        super(BaslerCamera, self).__init__()

        if pylon_camera is None:
            device = pylon.TlFactory.GetInstance().CreateFirstDevice()
            self.camera = pylon.InstantCamera(device)
        else:
            self.camera = pylon_camera

        self.model = self.camera.GetDeviceInfo().GetModelName()
        self.mono = self.model[-1] == 'm'
        self.exposure_time = exposure_time
        self.is_grabbing = False
        self.trigger_mode = trigger_mode

        self.camera_event_handler = CameraEventHandler(self.grab_result_callback)
        self.camera.RegisterImageEventHandler(self.camera_event_handler,\
            pylon.RegistrationMode_Append, pylon.Cleanup_Delete)


    def start_grabbing(self):

        print('BaslerCamera.start_grabbing')

        self.is_grabbing = True

        self.camera.MaxNumBuffer = 15

        self.camera.Open()

        self.camera.CenterX.SetValue(True)
        self.camera.CenterY.SetValue(True)
        self.camera.Width.SetValue(848)
        self.camera.Height.SetValue(848)

        self.camera.TriggerMode.SetValue('On' if self.trigger_mode else 'Off')
        self.camera.ExposureTime.SetValue(self.exposure_time)
        self.camera.Gain.SetValue(10.0)

        if not self.mono:
            # self.camera.PixelFormat.SetValue('RGB8')      # slower (because of computer or camera??)
            self.camera.PixelFormat.SetValue('BayerRG8')  # faster (but need debayering)

        self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)

    def stop_grabbing(self):
        self.is_grabbing = False
        self.camera.StopGrabbing()
        self.camera.Close()

    def grab_result_callback(self, grab_result):
        # print('grab_result_callback')
        if self.mono:
            img = grab_result.GetArray().copy()
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)
        else:
            img = cv2.cvtColor(grab_result.GetArray(), cv2.COLOR_BAYER_RG2BGR)

#        frame_number = grab_result.ImageNumber
#        print('FrameNr = {}'.format(frame_number))
#        print('img.shape = {}'.format(img.shape))

        timestamp = grab_result.TimeStamp
        frame = Frame(timestamp=timestamp, image=img)
        self.emit(frame)


class CameraEventHandler(pylon.ImageEventHandler):
    def __init__(self, handler, *args, **kwargs):
        super(CameraEventHandler, self).__init__(*args, **kwargs)
        self.handler = handler

    def OnImagesSkipped(self, camera, countOfSkippedImages):
        print("OnImagesSkipped event for device ", camera.GetDeviceInfo().GetModelName())
        print(countOfSkippedImages, " images have been skipped.")
        print('')

    def OnImageGrabbed(self, camera, grabResult):
        if grabResult.GrabSucceeded():
            self.handler(grabResult)
        else:
            print("Error: ", grabResult.GetErrorCode(), grabResult.GetErrorDescription())
        grabResult.Release()
