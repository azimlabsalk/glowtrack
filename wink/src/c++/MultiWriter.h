#ifndef MULTI_WRITER_H
#define MULTI_WRITER_H

#include <memory>
#include <queue>
#include <vector>

#include <pylon/PylonIncludes.h>

#include "Frame.h"
#include "Pipeline.h"

using namespace std;
using namespace Pylon;

typedef queue<shared_ptr<Frame>> frame_queue;
typedef vector<shared_ptr<CVideoWriter>> writer_group;

class MultiWriter : public Consumer<Frame> {
  protected:
    frame_queue writeQueue;
    std::shared_ptr<writer_group> videoWriters;
    std::shared_ptr<std::thread> writeThread;
  public:
    void start_writers();
    void attach_writers(std::shared_ptr<writer_group> writers);
    virtual void consume(std::shared_ptr<Frame> data);
    void wait();
};

#endif
