#include <iostream>
#include <string>
#include <utility>

#include <SWT.hpp>
#include <SWTInternal.hpp>

using namespace std;
using namespace cv;

SWT::SWT()
{
}

SWT::~SWT()
{
}

void SWT::getTransform(IplImage *pSrcImage,
                       IplImage *pSWTImage,
                       double lowThreshold,
                       double highThreshold,
                       int angleDivisor,
                       int bright_dark_text)
{
  SWTInternal swt;

  bool dark_on_light;

  if (bright_dark_text == 0)
    dark_on_light = false;
  else;
    dark_on_light = true;

  swt.getTransform(pSrcImage,
                   pSWTImage,
                   dark_on_light,
                   lowThreshold,
                   highThreshold,
                   angleDivisor);
}

