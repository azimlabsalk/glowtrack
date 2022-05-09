#ifndef TUPLE_PAIRER_H
#define TUPLE_PAIRER_H

#include <vector>
#include <memory>

#include "Pipeline.h"
#include "Frame.h"
#include "TuplePair.h"

using namespace std;

typedef vector<shared_ptr<Frame>> frame_tuple;

class TuplePairer : public PipelineStage<frame_tuple, TuplePair>  {
  protected:
    shared_ptr<frame_tuple> lastTuple;
  public:
    virtual void consume(std::shared_ptr<frame_tuple> data);
};


#endif
