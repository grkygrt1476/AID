# SPDX-FileCopyrightText: Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

CUDA_VER ?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CXX := g++
SRCS := gstdskltassist.cpp
INCS := $(wildcard *.h)
LIB := libnvdsgst_dskltassist.so

NVDS_VERSION := 8.0
GST_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS += -fPIC -DDS_VERSION=\"8.0.0\" \
	-I /usr/local/cuda-$(CUDA_VER)/include \
	-I /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/includes \
	-I /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/includes

PKGS := gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 opencv4
CFLAGS += $(shell pkg-config --cflags $(PKGS))

LIBS := -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -ldl \
	-lnppc -lnppig -lnpps -lnppicc -lnppidei \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta -lnvbufsurface -lnvbufsurftransform \
	-Wl,-rpath,$(LIB_INSTALL_DIR)
LIBS += $(shell pkg-config --libs $(PKGS))

OBJS := $(SRCS:.cpp=.o)

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
