MODULE_TOPDIR = $(shell grass --config path)

PGM = r.gpm.imerg

include $(MODULE_TOPDIR)/include/Make/Script.make

default: script
