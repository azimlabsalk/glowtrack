#ifndef TUPLE_WRITER_H
#define TUPLE_WRITER_H

#include <memory>
#include <queue>
#include <vector>
#include <thread>

#include <pylon/PylonIncludes.h>

#include "Frame.h"
#include "Pipeline.h"
#include "TuplePair.h"
#include "ThreadedVideoWriter.h"

using namespace std;
using namespace Pylon;

struct VideoInfo {
  uint32_t width;
  uint32_t height;
  EPixelType pixelType;
  int cFramesPerSecond;
  uint32_t cQuality;
};

typedef vector<shared_ptr<Frame>> frame_tuple;
typedef queue<shared_ptr<TuplePair>> tuple_queue;
typedef vector<shared_ptr<VideoInfo>> video_info_vector;
typedef vector<shared_ptr<ThreadedVideoWriter>> writer_group;
typedef vector<shared_ptr<std::ofstream>> ofstream_group;

class TupleWriter : public Consumer<TuplePair> {
  protected:
    std::string basepath;
    tuple_queue writeQueue;
    bool quitOnEmpty;
    int clipCount;
    std::shared_ptr<video_info_vector> videoInfos;
    std::shared_ptr<writer_group> videoWriters;
    std::shared_ptr<std::thread> writeThread;
    std::shared_ptr<ofstream_group> textWriters;
    void initWriters();
  public:
    TupleWriter(std::string path, std::shared_ptr<video_info_vector> videoInfoVec);
    virtual void consume(std::shared_ptr<TuplePair> data);
    void startNewClip();
    void endClip();
    void addTupleToClip(std::shared_ptr<TuplePair> tup);
    void wait();
    void worker();
    void startWriting();
};

#endif
