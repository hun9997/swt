#!/bin/bash

mkdir -p m4
if [ ! -f m4/ax_boost_log.m4 ]; then
    wget "http://git.savannah.gnu.org/gitweb/?p=autoconf-archive.git;a=blob_plain;f=m4/ax_boost_log.m4" -O m4/ax_boost_log.m4
fi
if [ ! -f m4/ax_boost_base.m4 ]; then
    wget "http://git.savannah.gnu.org/gitweb/?p=autoconf-archive.git;a=blob_plain;f=m4/ax_boost_base.m4" -O m4/ax_boost_base.m4
fi
if [ ! -f m4/ax_prog_doxygen.m4 ]; then
    wget "http://git.savannah.gnu.org/gitweb/?p=autoconf-archive.git;a=blob_plain;f=m4/ax_prog_doxygen.m4" -O m4/ax_prog_doxygen.m4
fi
if [ ! -f NEWS ]; then
    touch NEWS
fi
if [ ! -f COPYING ]; then
    touch COPYING
fi
if [ ! -f README ]; then
    touch README
fi
if [ ! -f AUTHORS ]; then
    touch AUTHORS
fi
if [ ! -f ChangeLog ]; then
    touch ChangeLog
fi

autoreconf -i -I m4
