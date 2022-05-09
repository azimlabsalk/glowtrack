#ifndef TUPLE_BUFFER_H
#define TUPLE_BUFFER_H

#include <vector>
#include <queue>

#include "Pipeline.h"
#include "TuplePair.h"
#include "Frame.h"

using namespace std;

class TupleBuffer : public PipelineStage<TuplePair, TuplePair>  {
  protected:
    queue<shared_ptr<TuplePair>> tupleQueue;
    int bufferSize;
    int tupleSize;
    bool bufferModeOn;
  public:
    TupleBuffer(int bufSize, int tupSize);
    virtual void consume(std::shared_ptr<TuplePair> data);
    void passThroughMode();
    void bufferingMode();
    void emitTerminator();
    void setBufferSize(int s);
};

#endif
