// ImageUtils.h
// Image utility functions
//

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

IplImage* ResizeImage(IplImage *pSrcImage,
                      int resizedHeight,
                      bool isBinaryImage = false);

IplImage* RestoreMask(IplImage *pMaskImage);

bool DoRectsOverlap(cv::Rect &rect1, cv::Rect &rect2);
