#!/bin/bash
if [ -f Makefile ]; then
    make clean
    make distclean
fi
rm -rf aclocal.m4 AUTHORS INSTALL config.* configure install-sh ltmain.sh Makefile.in missing NEWS depcomp autom4te.cache
