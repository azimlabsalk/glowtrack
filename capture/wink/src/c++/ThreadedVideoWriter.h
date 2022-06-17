#ifndef THREADED_VIDEO_WRITER_H
#define THREADED_VIDEO_WRITER_H

#include <memory>
#include <thread>
#include <queue>

#include <pylon/PylonIncludes.h>

#include "Frame.h"

using namespace std;
using namespace Pylon;

typedef queue<shared_ptr<Frame>> frame_queue;

class ThreadedVideoWriter {
  protected:
    bool quitOnEmpty = false;
    frame_queue frameQueue;
    CVideoWriter videoWriter;
    std::shared_ptr<std::thread> writeThread;
  public:
    void startWriting();
    void wait();
    void worker();
    void push(shared_ptr<Frame> frame);
    void SetParameter(uint32_t width, uint32_t height,
      EPixelType inputPixelType, double framesPerSecondPlaybackSpeed,
      uint32_t quality);
    void Open(const Pylon::String_t &filename);
    bool IsOpen() const;
    void Close();
};

#endif
