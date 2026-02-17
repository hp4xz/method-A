vpath %.cpp src/
vpath %.hh inc/

# ROOT libraries for drawing + fitting 
ROOTCFLAGS  = $(shell root-config --cflags)
ROOTLIBS = $(shell root-config --libs)  -lMinuit -lSpectrum
ROOTGLIBS = $(shell root-config --glibs)

CXX = g++
CPPFLAGS = -I inc/ -D_FILE_OFFSET_BITS=64 -fPIC
LDFLAGS = -L/usr/local/lib -L/usr/include -L/root/lib


MAIN = methodA
SOURCES = methodAFit.cpp
OBJECTS = $(addprefix bin/,$(SOURCES:.cpp=.o)) bin/$(MAIN).o
SOBJECT = $(addprefix bin/,$(SOURCES:.cpp=.so))
#OBJECTS = $(addprefix bin/,$(SOURCES:.cpp=.o))
INCLUDES = $(SOURCES:.cpp=.hh)

SHAREFLAGS = -shared -Wl,-soname,$(SOBJECT)

all: $(MAIN)

$(MAIN): $(OBJECTS) 
	$(CXX) $(CPPFLAGS) $(LDFLAGS) $(ROOTCFLAGS) -o $@ $(OBJECTS) $(ROOTLIBS) $(ROOTGLIBS) -std=c++17

bin/$(MAIN).o : $(MAIN).cpp $(INCLUDES)
	$(CXX) $(CPPFLAGS) $(LDFLAGS) $(ROOTCFLAGS) -c $< -o $@ $(ROOTLIBS) $(ROOTGLIBS) -std=c++17
	$(CXX) $(LDFLAGS) $(SHAREFLAGS) $(ROOTCFLAGS) -o $(SOBJECT) $(OBJECTS)
#bin/$(MAIN).so: $(SOBJECT)
	#$(CXX) $(SHAREFLAGS) -o bin/$(MAIN).o bin/$(MAIN).so

bin/%.o : %.cpp %.hh 
	$(CXX) $(CPPFLAGS) $(LDFLAGS) $(ROOTCFLAGS) -c $< -o $@ $(ROOTLIBS) $(ROOTGLIBS) -std=c++17

clean :
	rm -f bin/*.o bin/*.so *~ $(MAIN)

.PHONY : all clean
