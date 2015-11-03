#include "SWTLabeller.hpp"
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// #define __SWT_LABELLER_DISPLAY

using namespace cv;
using namespace gnim;
using namespace papyrus;

SWTLabeller::SWTLabeller()
{
}

SWTLabeller::~SWTLabeller()
{
}

gnim::Image SWTLabeller::operator()(const gnim::Image &image)
{
  IplImage *pInputImageIpl;
  IplImage *pOutputImageCharIpl;
  CvSize inputSize;
  Image *pOutputImageGNIM;
  int lightDarkText;
  vector<Rect> outWordBoxes;

  // Setup

  lightDarkText = 1;
  outWordBoxes.clear();

  // Convert image from gnim format to IplImage

  GNIMToIpl(&image, &pInputImageIpl);

  // Create output image

  inputSize.width  = pInputImageIpl->width;
  inputSize.height = pInputImageIpl->height;

  pOutputImageCharIpl  = cvCreateImage(inputSize, IPL_DEPTH_8U,  1);
  cvZero(pOutputImageCharIpl);

  // Processing

  findWords.process(pInputImageIpl, 
                    lightDarkText,
                    pOutputImageCharIpl,
                    outWordBoxes);
  
  IplToGNIM(pOutputImageCharIpl, &pOutputImageGNIM);

#ifdef __SWT_LABELLER_DISPLAY
  // Display the output image

  IplImage *pNormImageCharIpl;
  pNormImageCharIpl    = cvCreateImage(inputSize, IPL_DEPTH_8U,  1);
  Normalize(pOutputImageCharIpl, pNormImageCharIpl);

  cvNamedWindow("SWTLabeller: input image");
  cvShowImage("SWTLabeller: input image", pInputImageIpl);
  cvNamedWindow("SWTLabeller: labelled image");
  cvShowImage("SWTLabeller: labelled image", pNormImageCharIpl);
  cvWaitKey(0);
  cvReleaseImage(&pNormImageCharIpl);
#endif

  // Cleanup

  cvReleaseImage(&pInputImageIpl);
  cvReleaseImage(&pOutputImageCharIpl);

  // Return

  return *pOutputImageGNIM;
}
