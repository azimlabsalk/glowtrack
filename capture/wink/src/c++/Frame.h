#ifndef FRAME_H
#define FRAME_H

#include <pylon/PylonIncludes.h>

struct Frame {
  Pylon::CPylonImage image;
  uint32_t camera_context;
  bool video_terminator;
};

#endif
