#ifndef SKIPPER_H
#define SKIPPER_H

#include <vector>
#include <queue>
#include <memory>

#include "Pipeline.h"
#include "TuplePair.h"
#include "Frame.h"

using namespace std;

template <class T>
class Skipper : public PipelineStage<T, T>  {
  protected:
    int nSkipped;
    int nEmitted;
    int nth;
    int toEmit;
  public:
    Skipper(int nth, int emitK);
    virtual void consume(std::shared_ptr<T> data);
};

#endif
