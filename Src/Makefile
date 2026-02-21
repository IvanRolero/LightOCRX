# Compiler and flags
CXX = g++
CXXFLAGS = -std=gnu++17
INCLUDES = -I Libharu/src -I Libharu/include

# Libraries
LIBS = -L. -lonnxruntime -lhpdf -lpng -ljpeg -lz
STATIC_LIBS = libonnxruntime.a libhpdf.a

# Source files
SOURCES = main.cpp clipper.cpp text_detect.cpp text_recon.cpp pdf_export.cpp stb_impl.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Output executable
TARGET = lightocrx.exe

# Default target
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJECTS) $(STATIC_LIBS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET) $(LIBS)

# Compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJECTS) $(TARGET)

# Phony targets

.PHONY: all clean
