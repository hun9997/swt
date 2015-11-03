#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "ConvertUtils.h"

class SWT
{
 public:

  SWT();

  ~SWT();

  void getTransform(IplImage *pSrcImage,
                    IplImage *pSWTImage,
                    double lowThreshold,
                    double highThreshold,
                    int angleDivisor,
                    int bright_dark_text);

 private:

};
