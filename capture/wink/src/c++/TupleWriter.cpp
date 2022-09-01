#include <memory>
#include <thread>
#include <cassert>
#include <chrono>
#include <fstream>

#include "TupleWriter.h"

void TupleWriter::worker() {

  cout << "Writing images to files..." << endl;

  while (!writeQueue.empty() || !quitOnEmpty) {

    if (!writeQueue.empty()) {

      // cout << "Writing frame" << endl;

      shared_ptr<TuplePair> tup = writeQueue.front();
      writeQueue.pop();

      // cout << "writing tuple of size " << tup->frame_tuple1->size() << endl;
      // assert(tup->frame_tuple1->size() == videoWriters->size());

      bool clipDone = tup->frame_tuple1->at(0)->video_terminator;
      if (clipDone) {
        endClip();
      }
      else {
        bool clipOpen = videoWriters->at(0)->IsOpen();
        if (clipOpen) {
          addTupleToClip(tup);
        }
        else {
          startNewClip();
        }
      }
    }

  }

}

TupleWriter::TupleWriter(std::string path,
                         std::shared_ptr<video_info_vector> videoInfoVec) {
  basepath = path;
  quitOnEmpty = false;
  clipCount = 0;
  videoInfos = videoInfoVec;
  videoWriters = std::shared_ptr<writer_group>(new writer_group);
  textWriters = std::shared_ptr<ofstream_group>(new ofstream_group);
  initWriters();
}

void TupleWriter::startWriting() {
  writeThread = std::make_shared<std::thread>(&TupleWriter::worker, this);
}

void TupleWriter::initWriters() {

  for (int i = 0; i < videoInfos->size(); ++i) {

    shared_ptr<ThreadedVideoWriter> videoWriter(new ThreadedVideoWriter());
    shared_ptr<VideoInfo> info = videoInfos->at(i);

    cout << "setting params for writer " << i << endl;
    videoWriter->SetParameter(
        info->width,
        info->height,
        info->pixelType,
        info->cFramesPerSecond,
        info->cQuality );

    videoWriters->push_back(videoWriter);

    shared_ptr<std::ofstream> textStream(new std::ofstream);
    textWriters->push_back(textStream);
  }

}

void TupleWriter::startNewClip() {
  for (int i = 0; i < videoWriters->size(); ++i) {

    // Compute path
    std::ostringstream pathstream;
    pathstream << basepath << "/clip" << clipCount << "/cam" << i;

    std::ostringstream cmdstream;
    cmdstream << "mkdir -p " << pathstream.str();

    system(cmdstream.str().c_str());

    pathstream << "/interlaced.mp4";
    std::string videoPath = pathstream.str();

    // Open the video writer.
    videoWriters->at(i)->Open(videoPath.c_str());
    videoWriters->at(i)->startWriting();

    // Open the text writer
    std::ostringstream textPath;
    textPath << basepath << "/clip" << clipCount << "/cam" << i << "/timestamps.txt";
    textWriters->at(i)->open(textPath.str(), std::ios_base::out);

  }
  ++clipCount;
}

void TupleWriter::endClip() {
  for (int i = 0; i < videoWriters->size(); ++i) {
    videoWriters->at(i)->wait();
    videoWriters->at(i)->Close();
    textWriters->at(i)->close();
  }
}

void TupleWriter::addTupleToClip(std::shared_ptr<TuplePair> tup) {
  for (int i = 0; i < tup->frame_tuple1->size(); ++i) {
    std::shared_ptr<Frame> frame1 = tup->frame_tuple1->at(i);
    std::shared_ptr<Frame> frame2 = tup->frame_tuple2->at(i);
    videoWriters->at(frame1->camera_context)->push(frame1);
    videoWriters->at(frame2->camera_context)->push(frame2);

    (*(textWriters->at(i)))
       << frame1->camera_context
       << " " << frame1->timestamp
       << " " << frame1->image_number
       << " " << 1
       << endl;

    (*(textWriters->at(i)))
       << frame2->camera_context
       << " " << frame2->timestamp
       << " " << frame2->image_number
       << " " << 2
       << endl;

  }
}

void TupleWriter::consume(std::shared_ptr<TuplePair> data) {
  writeQueue.push(data);
}

void TupleWriter::wait() {
  quitOnEmpty = true;
  writeThread->join();
}
