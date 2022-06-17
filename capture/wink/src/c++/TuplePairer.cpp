#include "TuplePairer.h"

void TuplePairer::consume(std::shared_ptr<frame_tuple> data) {

  if (lastTuple.get() == nullptr) {

    lastTuple = data;

  } else {

    shared_ptr<TuplePair> tupPair(new TuplePair);
    tupPair->frame_tuple1 = lastTuple;
    tupPair->frame_tuple2 = data;

    lastTuple = nullptr;

    emit(tupPair);

  }

}
