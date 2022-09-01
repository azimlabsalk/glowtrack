#include "TuplePairer.h"

void TuplePairer::consume(std::shared_ptr<frame_tuple> data) {

  if (lastTuple.get() == nullptr) {

    lastTuple = data;

  } else {

    bool matched = true;

    for (int i = 0; i < lastTuple->size(); ++i) {
	uint64_t diff = data->at(i)->timestamp - lastTuple->at(i)->timestamp;
	matched = matched && (diff < 5050000) && (diff > 4950000);
    }

    if (matched) {
      shared_ptr<TuplePair> tupPair(new TuplePair);
      tupPair->frame_tuple1 = lastTuple;
      tupPair->frame_tuple2 = data;

      lastTuple = nullptr;

      emit(tupPair);
    } else {
      lastTuple = data;
    }

  }

}
