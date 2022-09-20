// Grab_MultipleCameras.cpp

/*

This script is partly based on sample code of the same name included with the Basler Pylon 5 C++ Samples.

*/

/* Note from the original Basler code sample:

    Note: Before getting started, Basler recommends reading the "Programmer's Guide" topic
    in the pylon C++ API documentation delivered with pylon.
    If you are upgrading to a higher major version of pylon, Basler also
    strongly recommends reading the "Migrating from Previous Versions" topic in the pylon C++ API documentation.

    This sample illustrates how to grab and process images from multiple cameras
    using the CInstantCameraArray class. The CInstantCameraArray class represents
    an array of instant camera objects. It provides almost the same interface
    as the instant camera for grabbing.
    The main purpose of the CInstantCameraArray is to simplify waiting for images and
    camera events of multiple cameras in one thread. This is done by providing a single
    RetrieveResult method for all cameras in the array.
    Alternatively, the grabbing can be started using the internal grab loop threads
    of all cameras in the CInstantCameraArray. The grabbed images can then be processed by one or more
    image event handlers. Please note that this is not shown in this example.
*/

#include <fstream>
#include <thread>
#include <queue>
#include <chrono>
#include <cstdlib>

// Include files to use the pylon API.
#include <pylon/PylonIncludes.h>
#include <pylon/usb/BaslerUsbInstantCamera.h>
#include <pylon/usb/BaslerUsbInstantCameraArray.h>

#include "Frame.h"
#include "Collator.h"
#include "TupleBuffer.h"
#include "TupleWriter.h"
#include "TuplePairer.h"
#include "MultiCamera.h"
#include "Skipper.h"
#include "Skipper.cpp"


using namespace std;
using namespace Pylon;
using namespace Basler_UsbCameraParams;

static const size_t c_maxCamerasToUse = 8;
static const uint32_t c_countOfImagesToGrab = 800 * c_maxCamerasToUse;
// static const int everyNth = 1;
// static const int bufferSize = 100;

// CPylonImage images[c_countOfImagesToGrab];
// int cam_contexts[c_countOfImagesToGrab];

typedef queue<shared_ptr<Frame>> frame_queue;
typedef vector<shared_ptr<ThreadedVideoWriter>> writer_group;


int main(int argc, char* argv[])
{
    // The exit code of the sample application.
    int exitCode = 0;

    // This is strangely needed to prevent "too many open files" errors.
    system("ulimit -n 4096");

    std::string data_dir(argv[1]);

    int everyNth = atoi(argv[2]);
    int bufferSize = atoi(argv[3]);
    int emitK = atoi(argv[4]);

    std::ostringstream cmdstream;
    cmdstream << "mkdir -p " << data_dir;

    system(cmdstream.str().c_str());

    // Before using any pylon methods, the pylon runtime must be initialized.
    PylonInitialize();

    try
    {

        // Get the transport layer factory.
        CTlFactory& tlFactory = CTlFactory::GetInstance();

        // Get all attached devices and exit application if no device is found.
        DeviceInfoList_t devices;
        if ( tlFactory.EnumerateDevices(devices) == 0 )
        {
            throw RUNTIME_EXCEPTION( "No camera present.");
        }

        // Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
        CBaslerUsbInstantCameraArray cameras( min( devices.size(), c_maxCamerasToUse));

        cout << "Got CBaslerUsbInstantCameraArray : size = " << cameras.GetSize() << endl;

        // Create and attach all Pylon Devices.
        for ( size_t i = 0; i < cameras.GetSize(); ++i)
        {
            cameras[ i ].Attach( tlFactory.CreateDevice( devices[ i ]));
        }

        cout << "Attached cameras" << endl;
        // Create and attach all Pylon Devices.
        std::shared_ptr<video_info_vector> videoInfos(new video_info_vector);
        for ( size_t i = 0; i < cameras.GetSize(); ++i)
        {
            CBaslerUsbInstantCamera& camera = cameras[i];

            // Print the model name of the camera.
            cout << "Using device " << camera.GetDeviceInfo().GetModelName() << endl;

            camera.MaxNumBuffer.SetValue(cameras.GetSize() * 8);

            // Open the camera.
            camera.Open();

            cout << "GrabLoopThreadPriority: " <<  camera.GrabLoopThreadPriority.GetValue()
            << endl;

            cout << "InternalGrabEngineThreadPriority: "
                 <<  camera.InternalGrabEngineThreadPriority.GetValue() << endl;


            camera.SequencerMode.SetValue(SequencerMode_Off);
            camera.SequencerConfigurationMode.SetValue(SequencerConfigurationMode_On);

            // camera.InternalGrabEngineThreadPriorityOverride = true;
            // camera.InternalGrabEngineThreadPriority.SetValue(25);
            //
            // camera.GrabLoopThreadPriorityOverride = true;
            // camera.GrabLoopThreadPriority.SetValue(24);

            camera.CenterX.SetValue(true);
            camera.CenterY.SetValue(true);
            camera.Width.SetValue(848);
            camera.Height.SetValue(848);

	    camera.TriggerSelector.SetValue(TriggerSelector_FrameStart);
            camera.TriggerMode.SetValue(TriggerMode_On);
            camera.TriggerSource.SetValue(TriggerSource_Line1);

	    camera.ExposureMode.SetValue(ExposureMode_Timed);
            camera.ExposureTime.SetValue(1250.0);

            // set up sequencer sets with different gains
            cout << "creating sequencer sets for camera " << i << endl;

            camera.Gain.SetValue(30.0);
            camera.SequencerSetSelector.SetValue(0);
            camera.SequencerSetSave.Execute();

            camera.Gain.SetValue(10.0);
            camera.SequencerSetSelector.SetValue(1);
            camera.SequencerSetSave.Execute();

            // set up sequencer paths (path 0 (reset) triggered by software, path 1 a loop (0-1-0-1-...))
            cout << "creating sequencer paths for camera " << i << endl;

            camera.SequencerSetSelector.SetValue(0);
            camera.SequencerSetLoad.Execute();
            camera.SequencerPathSelector.SetValue(0);
            camera.SequencerTriggerSource.SetValue(SequencerTriggerSource_SoftwareSignal1);
            camera.SequencerSetSave.Execute();

            camera.SequencerSetSelector.SetValue(0);
            camera.SequencerSetLoad();
            camera.SequencerPathSelector.SetValue(1);
            camera.SequencerTriggerSource.SetValue(SequencerTriggerSource_FrameStart);
            camera.SequencerSetSave.Execute();

            camera.SequencerSetSelector.SetValue(1);
            camera.SequencerSetLoad();
            camera.SequencerPathSelector.SetValue(1);
            camera.SequencerSetNext.SetValue(0);
            camera.SequencerSetSave.Execute();

            camera.SequencerMode.SetValue(SequencerMode_On);

            // reset the sequencer 
            camera.SoftwareSignalSelector.SetValue(SoftwareSignalSelector_SoftwareSignal1);
            camera.SoftwareSignalPulse.Execute();

            cout << "creating writer " << i << endl;

            // Get the required camera settings.
            CIntegerParameter width( camera.GetNodeMap(), "Width");
            CIntegerParameter height( camera.GetNodeMap(), "Height");
            CEnumParameter pixelFormat( camera.GetNodeMap(), "PixelFormat");

            // Map the pixelType
            CPixelTypeMapper pixelTypeMapper(&pixelFormat);
            EPixelType pixelType = pixelTypeMapper.GetPylonPixelTypeFromNodeValue(pixelFormat.GetIntValue());

            // The frame rate used for playing the video (playback frame rate).
            const int cFramesPerSecond = 30;
            // The quality used for compressing the video.
            const uint32_t cQuality = 100;

            // shared_ptr<CVideoWriter> videoWriter(new CVideoWriter());

            shared_ptr<VideoInfo> info = make_shared<VideoInfo>();
            info->width = (uint32_t) width.GetValue();
            info->height = (uint32_t) height.GetValue();
            info->pixelType = pixelType;
            info->cFramesPerSecond = cFramesPerSecond;
            info->cQuality = cQuality;

            videoInfos->push_back(info);

        }

        std::shared_ptr<MultiCamera> multi_cam(new MultiCamera(data_dir));
        std::shared_ptr<Collator> collator(new Collator(cameras.GetSize()));
        std::shared_ptr<TuplePairer> tuple_pairer(new TuplePairer);
        std::shared_ptr<TupleBuffer> tuple_buffer(new TupleBuffer(bufferSize, cameras.GetSize()));
        std::shared_ptr<Skipper<TuplePair>> skipper(new Skipper<TuplePair>(everyNth, emitK));
        std::shared_ptr<TupleWriter> tuple_writer(new TupleWriter(data_dir, videoInfos));


        // build the pipeline
        multi_cam->add_consumer(collator);
        collator->add_consumer(tuple_pairer);
        tuple_pairer->add_consumer(skipper);
        skipper->add_consumer(tuple_buffer);
        tuple_buffer->add_consumer(tuple_writer);

        // stop triggering
        //system("python -c 'from wink.triggers import ArduinoCameraTrigger; trigger = ArduinoCameraTrigger(); trigger.stop_triggering(); trigger.close()'");

        // grab frames
        cerr << endl << "Press Enter to begin triggering the lights and cameras." << endl;
        while( cin.get() != '\n');

        // system("/home/mnle/anaconda3/envs/glowtrack/bin/python3 -c 'from wink.triggers import CerebroCameraTrigger; trigger = CerebroCameraTrigger(); trigger.initialize(); import time; time.sleep(3)'");

        multi_cam->grab_frames(cameras);

        system("/home/mnle/anaconda3/envs/glowtrack/bin/python3 -c 'from wink.triggers import CerebroCameraTrigger; trigger = CerebroCameraTrigger(); trigger.initialize(); trigger.start_triggering()'");

        // grab frames
        cerr << endl << "Press Enter to begin capturing clips." << endl;
        while( cin.get() != '\n');

        tuple_writer->startWriting();

        string user_string;
        while (true) {

          cout << "start clip? (return = Y, q = quit)" << endl;
          getline(cin, user_string);
          if (user_string.length() > 0 && user_string[0] == 'q') {
            break;
          }

          tuple_buffer->passThroughMode();


          cout << "stop clip? (return = Y)" << endl;
          getline(cin, user_string);
          tuple_buffer->bufferingMode();
          tuple_buffer->emitTerminator();

          // else if (user_string.length() > 0 && user_string[0] == 'w') {
          //   cout << "writing" << endl;
          //   tuple_writer->startWriting();
          //
          //   cout << "wait until writer is done writing..." << endl;
          //   tuple_writer->wait();
          //   cout << "done." << endl;
          // }
        }

        cout << "stopping cameras" << endl;
        multi_cam->stop(cameras);

        // stop triggering
        system("/home/mnle/anaconda3/envs/glowtrack/bin/python3 -c 'from wink.triggers import CerebroCameraTrigger; trigger = CerebroCameraTrigger(); trigger.stop_triggering()'");
        //system("python -c 'from wink.triggers import ArduinoCameraTrigger; trigger = ArduinoCameraTrigger(); trigger.stop_triggering()'");

        cout << "writing" << endl;
//        tuple_writer->startWriting();

        cout << "wait until writer is done writing..." << endl;
        tuple_writer->wait();
        cout << "Done." << endl;

    }
    catch (const GenericException &e)
    {
        // Error handling
        cerr << "An exception occurred." << endl
        << e.GetDescription() << endl;
        exitCode = 1;
    }

    // Comment the following two lines to disable waiting on exit.
    cerr << endl << "Press Enter to exit." << endl;
    while( cin.get() != '\n');

    // Releases all pylon resources.
    PylonTerminate();

    return exitCode;
}
