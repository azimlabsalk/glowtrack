#ifndef PIPELINE_H
#define PIPELINE_H

#include <vector>
#include <memory>
#include <iostream>

template <class I>
class Consumer {
  public:
    virtual void consume(std::shared_ptr<I> data) {
      std::cerr << "'Consumer.consume()' should be overridden" << std::endl;
    };
};

template <class O>
class Producer {
  protected:
    std::vector<std::shared_ptr<Consumer<O>>> consumers;
  public:
    void add_consumer(std::shared_ptr<Consumer<O>> consumer) {
      consumers.push_back(consumer);
    }
    void emit(std::shared_ptr<O> data) {
      for (int i = 0; i < consumers.size(); ++i) {
        consumers[i]->consume(data);
      }
    }
};

template <class I, class O>
class PipelineStage: public Consumer<I>, public Producer<O>{};

#endif
