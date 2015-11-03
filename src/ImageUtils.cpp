// ImageUtils.cpp
//
// Image utility functions
//
// Peter Wendt
// April 15, 2014
// Zeitera LLC

#include <algorithm>
#include "ImageUtils.h"
#include "ShapeDetect.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

IplImage* ResizeImage(IplImage *pSrcImage,
                      int resizedHeight,
                      bool isBinaryImage)
{
  IplImage *pResizedImage;

  int inputWidth;
  int inputHeight;
  int outputWidth;
  int outputHeight;
  CvSize outputSize;
  double resizeFactor;
  int interpMethod;

  // Get input size

  inputWidth  = pSrcImage->width;
  inputHeight = pSrcImage->height;

  // Compute output size

  outputHeight = resizedHeight;

  resizeFactor = (double) outputHeight / (double) inputHeight;
  outputWidth  = (int) floor(resizeFactor * inputWidth + 0.5);

  outputSize.width  = outputWidth;
  outputSize.height = outputHeight;

  // Create resized IplImage

  pResizedImage = cvCreateImage(outputSize,
                                pSrcImage->depth,
                                pSrcImage->nChannels);

  if (pResizedImage == NULL)
    return NULL;

  // Select the interpolation method

   if (isBinaryImage)
    interpMethod = CV_INTER_NN;
   else if (outputHeight > inputHeight)
    interpMethod = CV_INTER_CUBIC;
  else
    interpMethod = CV_INTER_AREA;

  // Resize

  cvResize(pSrcImage, pResizedImage, interpMethod);

  return pResizedImage;
}

IplImage* RestoreMask(IplImage *pMaskImage)
{
  IplImage *pDstImage;
  CvSize    dstSize;
  CvPoint  *pVertices;
  int       numVertices;

  ShapeDetect shapeDetect;

  dstSize = cvSize(pMaskImage->width, pMaskImage->height);

  pDstImage = cvCreateImage(dstSize, pMaskImage->depth, pMaskImage->nChannels);

  if (pDstImage == NULL)
    return NULL;

  // Compute convex hull

  shapeDetect.ConvexHull(pMaskImage,
                         pDstImage,
                         &pVertices,
                         numVertices);

  delete[] pVertices;

  return pDstImage;
}

bool DoRectsOverlap(Rect &rect1, Rect &rect2)
{
  bool overlap;

  overlap = true;

  // First rect to right of second

  if (rect1.x >= (rect2.x + rect2.width))
    overlap = false;
  
  // Left

  if ((rect1.x + rect1.width) <= rect2.x)
    overlap = false;

  // Below

  if (rect1.y >= (rect2.y + rect2.height))
    overlap = false;

  // Above

  if ((rect1.y + rect1.height) <= rect2.y)
    overlap = false;

  return overlap;
}
