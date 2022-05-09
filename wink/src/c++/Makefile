OPENCV_CFLAGS=`pkg-config --cflags opencv`
OPENCV_LIBS=`pkg-config --libs opencv`

PYLON_CFLAGS=`/opt/pylon5/bin/pylon-config --cflags`
PYLON_LIBS=`/opt/pylon5/bin/pylon-config --libs`

IDIR =./
CC=g++
CFLAGS=-I$(IDIR)

ODIR=./build

_DEPS = Frame.h Pipeline.h MultiCamera.h MultiWriter.h TupleWriter.h Collator.h TupleBuffer.h TuplePairer.h ThreadedVideoWriter.h Skipper.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = Grab_MultipleCameras.o MultiCamera.o MultiWriter.o TupleWriter.o Collator.o TupleBuffer.o TuplePairer.o ThreadedVideoWriter.o Skipper.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) $(PYLON_CFLAGS) $(OPENCV_CFLAGS) -std=c++11 -DUSE_GIGE -c -o $@ $< $(CFLAGS) -lpthread

build/Grab_MultipleCameras: $(OBJ)
	$(CC) -Wl,--enable-new-dtags -Wl,-rpath,/opt/pylon5/lib64 -o $@ $^ $(CFLAGS) $(PYLON_LIBS) $(OPENCV_LIBS) -lpthread

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(IDIR)/*~
