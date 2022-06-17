#include <memory>
#include <thread>

#include "MultiWriter.h"

void write_worker(frame_queue* q, writer_group* writers) {
  cout << "Writing images to files..." << endl;
  while (true) {
    if (!q->empty()) {
      cout << "Writing frame" << endl;
      shared_ptr<Frame> frame;
      frame = q->front();
      q->pop();
      if (frame->video_terminator) {
        writers->at(frame->camera_context)->Close();
        bool allDone = true;
        for (int i = 0; i < writers->size(); ++i) {
          allDone = allDone && !writers->at(i)->IsOpen();
        }
        if (allDone) {
          break;
        }
      } else {
        writers->at(frame->camera_context)->Add(frame->image);
      }
    }
  }
}


void MultiWriter::attach_writers(std::shared_ptr<writer_group> writers) {
  videoWriters = writers;
}

void MultiWriter::consume(std::shared_ptr<Frame> data) {
  writeQueue.push(data);
}

void MultiWriter::start_writers() {
  // Note: call _after_ videoWriters is populated
  writeThread = std::make_shared<std::thread>(write_worker, &writeQueue, &(*videoWriters));
}

void MultiWriter::wait() {
  writeThread->join();
}
