#include "SWTBoxFinder.hpp"
#include "ConvertUtils.h"
#include "Process.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

// #define __SWT_BOXFINDER_DISPLAY

using namespace std;
using namespace cv;
using namespace gnim;
using namespace papyrus;

SWTBoxFinder::SWTBoxFinder()
{
}

SWTBoxFinder::~SWTBoxFinder()
{
}

void SWTBoxFinder::operator()(const gnim::Image& labelled_image,
                              papyrus::inserter<bounding_box_t> output)
{
  IplImage *pLabelledImageIpl;
  vector<Mat> blobMasks;
  vector<Rect> uprightBlobBoxes;
  vector<RotatedRect> minBlobBoxes;
  bounding_box_t tempBox;
  int i;
  int j;
  int listSize;

  // Init

  blobMasks.clear();
  uprightBlobBoxes.clear();
  minBlobBoxes.clear();

  // Convert image from gnim format to IplImage

  GNIMToIpl(&labelled_image, &pLabelledImageIpl);

  // Find upright bounding boxes and masks

#if 0
  FindBoxesAndMasks(pLabelledImageIpl,
                    uprightBlobBoxes,
                    blobMasks);
#else
  FindBoxes(pLabelledImageIpl,
            uprightBlobBoxes);

  GetBlobMasks(pLabelledImageIpl,
               uprightBlobBoxes,
               blobMasks);
#endif

  listSize = uprightBlobBoxes.size();

  // Find minimal bounding boxes

  FindMinBoxes(blobMasks,
               uprightBlobBoxes,
               minBlobBoxes);

  // Convert output RotatedRect boxes to papyrus bounding_box_t boxes
  // Currently, the inserter inserts at the front of the list, not the
  // back, so we go through the boxes in reverse order

  for (i = 0; i < listSize; i++)
  {
    j = listSize - 1 - i;

    RotatedRectToBox(minBlobBoxes[j], tempBox);
    output.insert(tempBox);
  }

#ifdef __SWT_BOXFINDER_DISPLAY
  // Create and display box images

  IplImage *pTempImage;;
  IplImage *pBoxImageIpl;

  CvSize inputSize;

  // Setup

  inputSize.width  = pLabelledImageIpl->width;
  inputSize.height = pLabelledImageIpl->height;

  pTempImage   = cvCreateImage(inputSize, IPL_DEPTH_8U, 1);
  pBoxImageIpl = cvCreateImage(inputSize, IPL_DEPTH_8U, 3);

  Outline(pLabelledImageIpl, pTempImage);
  cvCvtColor(pTempImage, pBoxImageIpl, CV_GRAY2RGB);

  // Draw the boxes

  DrawBoxes(pBoxImageIpl, uprightBlobBoxes);
  DrawMinBoxes(pBoxImageIpl, minBlobBoxes, 1);

  // Display
  
  cvNamedWindow("SWTBoxFinder: box image");
  cvShowImage("SWTBoxFinder: box image", pBoxImageIpl);
  cvWaitKey(0);

  // Cleanup

  cvReleaseImage(&pTempImage);
  cvReleaseImage(&pBoxImageIpl);
#endif

  // Cleanup

  cvReleaseImage(&pLabelledImageIpl);

  // Return

  return;
}
