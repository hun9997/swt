ACLOCAL_AMFLAGS = -I m4
AM_CFLAGS = $(CFLAGS) -I. -I$(top_srcdir)/include/swt -I/usr/local/include/opencv2 $(SIMD_FLAGS)
AM_CXXFLAGS = $(CXXFLAGS) $(CPPFLAGS) -I. -I$(top_srcdir)/include/swt -I/usr/local/include/opencv2 $(SIMD_FLAGS) -stdlib=libc++ -std=c++11
lib_LTLIBRARIES = libswt.la
libswt_la_SOURCES = ConvertUtils.cpp \
                    FindWords.cpp \
                    ImageUtils.cpp \
                    Process.cpp \
                    Rotate.cpp \
                    ShapeDetect.cpp \
                    SWT.cpp \
                    SWTInternal.cpp \
                    SWTBoxFinder.cpp \
                    SWTExtractor.cpp \
                    SWTLabeller.cpp \
                    TextConfig.cpp \
                    TextDetect.cpp \
                    Timer.cpp
libswt_la_LDFLAGS = -L$(top_srcdir)/src  -L/usr/local/lib -version-info 0:0:0 -export-dynamic -no-undefined
libswt_la_LIBADD = $(LIBS) -lboost_filesystem -lboost_system -lopencv_core -lopencv_highgui -lopencv_imgproc -lgnim -lgnimpp -lpapyrus -ltesseract
