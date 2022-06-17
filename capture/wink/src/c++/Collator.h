#ifndef COLLATOR_H
#define COLLATOR_H

#include <vector>
#include <queue>
#include <memory>

#include "Pipeline.h"
#include "Frame.h"

using namespace std;

typedef vector<shared_ptr<Frame>> frame_tuple;
typedef queue<shared_ptr<Frame>> frame_queue;

class Collator : public PipelineStage<Frame, frame_tuple>  {
  protected:
    vector<shared_ptr<frame_queue>> queues;
  public:
    Collator(uint32_t nCams);
    virtual void consume(std::shared_ptr<Frame> data);
};

#endif
