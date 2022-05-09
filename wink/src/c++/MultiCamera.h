#ifndef MULTI_CAMERA_H
#define MULTI_CAMERA_H

#include <memory>
#include <queue>
#include <thread>
#include <vector>

#include <pylon/usb/BaslerUsbInstantCameraArray.h>

#include "Pipeline.h"
#include "Frame.h"

using namespace Pylon;

class MultiCamera : public Producer<Frame> {
  protected:
    std::shared_ptr<std::thread> grabThread;
    bool isGrabbing;
    std::string dataDir;
    std::string timestampFile;
  public:
    void grab_frames(uint32_t framesToGrab, CBaslerUsbInstantCameraArray& cameras);
    void grab_frames(CBaslerUsbInstantCameraArray& cameras);
    void worker(CBaslerUsbInstantCameraArray* camerasPtr);
    void stop(CBaslerUsbInstantCameraArray& cameras);
    MultiCamera(std::string data_dir);
};

#endif
