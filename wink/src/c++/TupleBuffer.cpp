#include "TupleBuffer.h"

#include <memory>

using namespace std;

shared_ptr<TuplePair> makeTerminatorTuplePair(int nCams) {
  shared_ptr<frame_tuple> tup1(new frame_tuple);
  shared_ptr<frame_tuple> tup2(new frame_tuple);
  for (int i = 0; i < nCams; ++i) {
    shared_ptr<Frame> frame(new Frame);
    frame->camera_context = i;
    frame->video_terminator = true;
    tup1->push_back(frame);
    tup2->push_back(frame);
  }
  shared_ptr<TuplePair> tupPair(new TuplePair);
  tupPair->frame_tuple1 = tup1;
  tupPair->frame_tuple2 = tup2;
  return tupPair;
}

TupleBuffer::TupleBuffer(int bufSize, int tupSize) {
  bufferSize = bufSize;
  tupleSize = tupSize;
  bufferModeOn = true;
}

void TupleBuffer::consume(std::shared_ptr<TuplePair> data) {
  if (bufferModeOn) {
    tupleQueue.push(data);
    if (tupleQueue.size() > bufferSize) {
      tupleQueue.pop();
    }
  } else {
    emit(data);
  }
};

void TupleBuffer::passThroughMode() {
  // THIS IS TECHNICALLY NOT THREAD SAFE, BUT SHOULD MOSTLY WORK
  while (!tupleQueue.empty()) {
    shared_ptr<TuplePair> tup = tupleQueue.front();
    tupleQueue.pop();
    emit(tup);
  }
  bufferModeOn = false;
};

void TupleBuffer::bufferingMode() {
  bufferModeOn = true;
};

void TupleBuffer::emitTerminator() {
  emit(makeTerminatorTuplePair(tupleSize));
};

void TupleBuffer::setBufferSize(int s) {
  bufferSize = s;
}
