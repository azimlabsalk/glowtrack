#include <memory>

#include "Collator.h"

using namespace std;

Collator::Collator(uint32_t nCams) {
  for (int i = 0; i < nCams; ++i) {
    queues.push_back(make_shared<frame_queue>());
  }
}

void Collator::consume(std::shared_ptr<Frame> data) {
  int cam_idx = data->camera_context;
  queues[cam_idx]->push(data);
  bool tup_ready = true;
  for (int i = 0; i < queues.size(); ++i) {
    tup_ready = tup_ready && !queues[i]->empty();
  }
  if (tup_ready) {
    shared_ptr<frame_tuple> tup(new frame_tuple);
    for (int i = 0; i < queues.size(); ++i) {
      tup->push_back(queues[i]->front());
      queues[i]->pop();
    }
    emit(tup);
  }
}
