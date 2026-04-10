UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
include makefile.OSX
else
include makefile
endif
