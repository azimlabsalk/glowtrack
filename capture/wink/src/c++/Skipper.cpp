#include "Skipper.h"

#include <memory>
#include <iostream>

using namespace std;

template <class T>
Skipper<T>::Skipper(int everyNth, int emitK) {
  assert(everyNth >= 1);
  assert(emitK >= 1);
  assert(emitK <= everyNth);
  nth = everyNth;
  toEmit = emitK;
  nSkipped = 0;
  nEmitted = 0;
};

template <class T>
void Skipper<T>::consume(std::shared_ptr<T> data) {
  if (nSkipped == nth - toEmit) {
    this->emit(data);
    ++nEmitted;
    if (nEmitted == toEmit) {
      nEmitted = 0;
      nSkipped = 0;
    }
  } else {
    ++nSkipped;
  }
};
