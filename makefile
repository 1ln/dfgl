CC=g++
CFLAGS=-c -Wall -std=c++11
LDFLAGS=
SOURCES=main.cpp Shader.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=prog
INCLUDE=Shader.h
LIBS = -L /usr/X11/lib/ -lGL -lglfw -lGLEW
all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
     $(CC) $(LDFLAGS) $(OBJECTS) -o $@ $(LIBS)

.cpp.o:
     $(CC) $(CFLAGS) $< -o $(INCLUDE) $@







