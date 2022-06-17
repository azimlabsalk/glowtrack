#ifndef TUPLE_PAIR_H
#define TUPLE_PAIR_H

#include <vector>
#include <memory>

#include "Frame.h"

using namespace std;

typedef vector<shared_ptr<Frame>> frame_tuple;

struct TuplePair {
  shared_ptr<frame_tuple> frame_tuple1;
  shared_ptr<frame_tuple> frame_tuple2;
};

#endif
