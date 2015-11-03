ACLOCAL_AMFLAGS = -I m4
AM_CFLAGS = $(CFLAGS) -I$(top_srcdir)/include -I$(top_srcdir)/include/swt -I/usr/local/include -I/usr/local/include/papyrus $(SIMD_FLAGS)
AM_CXXFLAGS = $(CXXFLAGS) $(CPPFLAGS) -I$(top_srcdir)/include -I$(top_srcdir)/include/swt -I/usr/local/include -I/usr/local/include/papyrus $(SIMD_FLAGS) -std=c++11 -stdlib=libc++

noinst_PROGRAMS = textExtract textExtractDir

textExtract_SOURCES = textExtract.cpp
textExtract_LDFLAGS = -L$(top_srcdir)/src -L/usr/local/lib
textExtract_LDADD = -lopencv_core -lopencv_imgproc -lopencv_highgui -lgnim -lgnimpp -lpapyrus -ltesseract -lswt

textExtractDir_SOURCES = textExtractDir.cpp
textExtractDir_LDFLAGS = -L$(top_srcdir)/src -L/usr/local/lib
textExtractDir_LDADD = -lopencv_core -lopencv_imgproc -lopencv_highgui -lgnim -lgnimpp -lpapyrus -ltesseract -lswt
