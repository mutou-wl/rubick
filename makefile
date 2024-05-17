CC := g++
SRCDIR := src

STD := -std=c++17

# SOURCES := $(shell find $(SRCDIR) -type f -name *.cpp)
# OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.cpp=.o))
# LIB_DIR = ~/TENET/external/lib
# INCLUDE_DIR = ~/TENET/external/include
LIB_DIR := /home/wcp/Project/DDPM_code/TENET/external/lib
INCLUDE_DIR := /home/wcp/Project/DDPM_code/TENET/external/include
LIB := -lbarvinok -lisl -lntl -lpolylibgmp -lgmp

memEst:
	@mkdir -p bin
	@$(CC) $(STD) -I ${INCLUDE_DIR} -L ${LIB_DIR} ${SRCDIR}/memEstimate.cpp -o bin/memEst $(LIB)

