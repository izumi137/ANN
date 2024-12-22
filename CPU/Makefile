# Compiler and flags
CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -Wall
NVCCFLAGS = -std=c++17

# Files and output
TARGET = CPU
CPP_SOURCES = basedata.cpp dataloader.cpp utils.cpp module.cpp crossentropy.cpp linear.cpp model.cpp relu.cpp softmax.cpp 
CU_SOURCES = CPU.cu
OBJECTS = $(CPP_SOURCES:.cpp=.o) $(CU_SOURCES:.cu=.o)

# Rules
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)
