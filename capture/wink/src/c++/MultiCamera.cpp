#include <iostream>
#include <fstream>

#include <pylon/PylonIncludes.h>
#include <pylon/ThreadPriority.h>

#include "MultiCamera.h"

using namespace std;
using namespace Pylon;

void MultiCamera::grab_frames(uint32_t framesToGrab, CBaslerUsbInstantCameraArray& cameras) {

  isGrabbing = true;
  cameras.StartGrabbing();

  std::ofstream outfile;
  outfile.open(timestampFile, std::ios_base::out); // append instead of overwrite

  // This smart pointer will receive the grab result data.
  CGrabResultPtr ptrGrabResult;

  // Grab c_countOfImagesToGrab from the cameras.
  for( uint32_t i = 0; i < framesToGrab && cameras.IsGrabbing(); ++i)
  {
      cameras.RetrieveResult( 5000, ptrGrabResult, TimeoutHandling_ThrowException);

      intptr_t cameraContextValue = ptrGrabResult->GetCameraContext();

      outfile << cameraContextValue << " " << ptrGrabResult->GetTimeStamp() << endl;

      shared_ptr<Frame> frame(new Frame);
      frame->image.CopyImage(ptrGrabResult);
      frame->camera_context = cameraContextValue;
      frame->video_terminator = false;
      emit(frame);

      // Print the index and the model name of the camera.
      cout << "Camera " <<  cameraContextValue << ": " << cameras[cameraContextValue].GetDeviceInfo().GetModelName() << endl;

  }

  // Terminate all the videos
  for (int cam_idx = 0; cam_idx < cameras.GetSize(); ++cam_idx) {
    shared_ptr<Frame> frame(new Frame);
    frame->camera_context = cam_idx;
    frame->video_terminator = true;
    emit(frame);
  }

}

void MultiCamera::grab_frames(CBaslerUsbInstantCameraArray& cameras) {

  grabThread = std::make_shared<std::thread>(&MultiCamera::worker, this, &cameras);

}

void MultiCamera::worker(CBaslerUsbInstantCameraArray* camerasPtr) {

  SetRTThreadPriority(GetCurrentThreadHandle(), 24);
  isGrabbing = true;

  CBaslerUsbInstantCameraArray &cameras = *camerasPtr;

  cameras.StartGrabbing();

  std::ofstream outfile;
  outfile.open(timestampFile, std::ios_base::out);

  // This smart pointer will receive the grab result data.
  CGrabResultPtr ptrGrabResult;

  // Grab c_countOfImagesToGrab from the cameras.
  while(isGrabbing) {
      cameras.RetrieveResult( 5000, ptrGrabResult, TimeoutHandling_ThrowException);

      intptr_t cameraContextValue = ptrGrabResult->GetCameraContext();

      outfile << cameraContextValue << " " << ptrGrabResult->GetTimeStamp() << endl;

      shared_ptr<Frame> frame(new Frame);
      frame->image.CopyImage(ptrGrabResult);
      frame->camera_context = cameraContextValue;
      frame->video_terminator = false;
      emit(frame);

      // Print the index and the model name of the camera.
      // cout << "Camera " <<  cameraContextValue << ": " << cameras[cameraContextValue].GetDeviceInfo().GetModelName() << endl;
  }

  cout << "Camera stage exiting." << endl;

  // // Terminate all the videos
  // for (int cam_idx = 0; cam_idx < cameras.GetSize(); ++cam_idx) {
  //   shared_ptr<Frame> frame(new Frame);
  //   frame->camera_context = cam_idx;
  //   frame->video_terminator = true;
  //   emit(frame);
  // }

}

MultiCamera::MultiCamera(std::string data_dir) {
  dataDir = data_dir;

  std::ostringstream ts;
  ts << dataDir << "/timestamps.txt";
  timestampFile = ts.str();
}

void MultiCamera::stop(CBaslerUsbInstantCameraArray& cameras) {
  isGrabbing = false;
  // cameras.StopGrabbing();
  grabThread->join();
}
