#ifndef FRAME_H
#define FRAME_H

#include <pylon/PylonIncludes.h>

struct Frame {
  Pylon::CPylonImage image;
  uint32_t camera_context;
  uint64_t timestamp;
  int64_t num_skipped_images;
  int64_t image_number;
  int64_t id;
  uint64_t block_id;
  bool video_terminator;
};

#endif
