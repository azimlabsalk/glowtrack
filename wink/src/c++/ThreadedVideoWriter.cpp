#include "ThreadedVideoWriter.h"

#include <pylon/PylonIncludes.h>

using namespace std;
using namespace Pylon;

void ThreadedVideoWriter::worker() {

  while (!frameQueue.empty() || !quitOnEmpty) {
    if (!frameQueue.empty()) {
      shared_ptr<Frame> frame = frameQueue.front();
      videoWriter.Add(frame->image);
      frameQueue.pop();
    }
    else {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

}

void ThreadedVideoWriter::wait() {
  quitOnEmpty = true;
  writeThread->join();
}

void ThreadedVideoWriter::startWriting() {
  quitOnEmpty = false;
  writeThread = std::make_shared<std::thread>(&ThreadedVideoWriter::worker, this);
}

void ThreadedVideoWriter::SetParameter(uint32_t width, uint32_t height,
  EPixelType inputPixelType, double framesPerSecondPlaybackSpeed,
  uint32_t quality) {

  videoWriter.SetParameter(width, height, inputPixelType,
    framesPerSecondPlaybackSpeed, quality);
}

void ThreadedVideoWriter::Open(const Pylon::String_t &filename) {
  videoWriter.Open(filename);
}

bool ThreadedVideoWriter::IsOpen() const {
  return videoWriter.IsOpen();
}

void ThreadedVideoWriter::Close() {
  videoWriter.Close();
}

void ThreadedVideoWriter::push(shared_ptr<Frame> frame) {
  frameQueue.push(frame);
}
