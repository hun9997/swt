#include <iostream>
#include <string>
#include <utility>

#include "Process.h"
#include "SWT.hpp"
#include "ShapeDetect.hpp"
#include "TextDetect.hpp"

using namespace std;
using namespace cv;

// #define __LOG_PERF_STATS
// #define __PRINT_PERF_STATS

TextDetect::TextDetect(TextConfig *textConfig)
{
  if (textConfig == NULL)
    SetDefaultConfig();
  else
    config = *textConfig;

  isInitialized = false;
}

TextDetect::~TextDetect()
{
  if (isInitialized == true)
    DeleteImages();
}

void TextDetect::Init(int imageHeight)
{
  numBlobsFound = 0;
}

void TextDetect::SetDefaultConfig()
{
  // Modes

  config.darkText      = true;
  config.lightText     = true;
  config.logSWT        = false;
  config.smoothSWT     = false;
  config.separateBlobs = false;
  config.mergeBlobs    = false;
  config.suppressEdges = false;
  config.padRegions    = false;
  config.suppressSmallRegions = false;
  config.suppressThickStrokes = true;
  config.displayImages = false;
  config.saveImage     = false;

  // Parameters

  config.maxNumBlobs = 10000;
  config.angleRangeDivisor = 2;
  config.minStrokeWidthDivisor = 50;
  config.maxStrokeWidthDivisor = 20;
  config.smoothFilterSize = 3;
  config.separateFilterSize = 1;
  config.mergeFilterSize  = 1;
  config.dilateRadius     = 2;
  config.padWidth         = 0;
  config.cannyLowThreshold  = 100;
  config.cannyHighThreshold = 250;
  config.fgMinValue      = 100;
  config.strokePixelDiff = 5;
}

void TextDetect::InitImages()
{
  int i;

  pEdgeImage    = cvCreateImage(imageSize, IPL_DEPTH_8U,  1);
  pTempImage    = cvCreateImage(imageSize, IPL_DEPTH_32F, 1);
  pTextBoxImage = cvCreateImage(imageSize, IPL_DEPTH_8U,  3);

  for (i = 0; i < 2; i++)
  {
    pSWTImage[i]     = cvCreateImage(imageSize, IPL_DEPTH_32F, 1);
    pSWTMaskImage[i] = cvCreateImage(imageSize, IPL_DEPTH_8U,  1);
    pBlobImage[i]    = cvCreateImage(imageSize, IPL_DEPTH_32F, 1);
  }

  if (config.displayImages)
  {
    for (i = 0; i < 2; i++)
    {
      pSWTDisplayImage[i]  = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
      pBlobDisplayImage[i] = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
    }
  }
}

void TextDetect::DeleteImages()
{
  int i;

  for (i = 0; i < 2; i++)
  {
    cvReleaseImage(&pSWTImage[i]);
    cvReleaseImage(&pSWTMaskImage[i]);
    cvReleaseImage(&pBlobImage[i]);
  }

  cvReleaseImage(&pEdgeImage);
  cvReleaseImage(&pTempImage);
  cvReleaseImage(&pTextBoxImage);

  if (config.displayImages)
  {
    for (i = 0; i < 2; i++)
    {
      cvReleaseImage(&pSWTDisplayImage[i]);
      cvReleaseImage(&pBlobDisplayImage[i]);
    }
  }
}

void TextDetect::InitWindows()
{
  cvNamedWindow("Input Image",           CV_WINDOW_AUTOSIZE);
  cvNamedWindow("SWT Bright Image",      CV_WINDOW_AUTOSIZE);
  cvNamedWindow("SWT Dark Image",        CV_WINDOW_AUTOSIZE);
  cvNamedWindow("SWT Bright Mask Image", CV_WINDOW_AUTOSIZE);
  cvNamedWindow("SWT Dark Mask Image",   CV_WINDOW_AUTOSIZE);
  cvNamedWindow("Blob Bright Image",     CV_WINDOW_AUTOSIZE);
  cvNamedWindow("Blob Dark Image",       CV_WINDOW_AUTOSIZE);
  cvNamedWindow("Text Box Image",        CV_WINDOW_AUTOSIZE);
}

void TextDetect::DisplayImages()
{
  cvShowImage("Input Image",           pInputImage);
  cvShowImage("Blob Bright Image",     pBlobDisplayImage[0]);
  cvShowImage("Blob Dark Image",       pBlobDisplayImage[1]);
  cvShowImage("SWT Bright Image",      pSWTDisplayImage[0]);
  cvShowImage("SWT Dark Image",        pSWTDisplayImage[1]);
  cvShowImage("SWT Bright Mask Image", pSWTMaskImage[0]);
  cvShowImage("SWT Dark Mask Image",   pSWTMaskImage[1]);
  cvShowImage("Text Box Image",        pTextBoxImage);

  cvWaitKey(0);
}

void TextDetect::CreateBoxImage(vector<Rect> &textBoxes)
{
  cvCvtColor(pInputImage, pTextBoxImage, CV_GRAY2RGB);
  DrawBoxes(pTextBoxImage, textBoxes);

  if (config.saveImage)
   cvSaveImage("box_image.jpg", pTextBoxImage);
}

void TextDetect::GetText(IplImage *pSrcImage,
                         vector<Mat>  &textMasks,
                         vector<Rect> &textBoxes)
{
  int i;
  int j;
  int minBoxSize;
  Mat blobMask;
  Rect blobRect;

  // Setup

  Init(pSrcImage->height);

  pBlobStats  = new CvConnectedComp[config.maxNumBlobs];
  pBlobSelect = new bool[config.maxNumBlobs];

  // Create images

  imageSize.width  = pSrcImage->width;
  imageSize.height = pSrcImage->height;

  if (isInitialized == true)
    DeleteImages();

  InitImages();

  isInitialized = true;

  pInputImage = pSrcImage;

  minBoxSize = (int) floor(imageSize.height / config.minStrokeWidthDivisor);
 
  if (config.suppressEdges)
  {
    // Create Canny edge image

    cvCanny(pSrcImage,
            pEdgeImage,
            config.cannyLowThreshold,
            config.cannyHighThreshold,
            3);
  }

  // Core computations

  // We compute two SWT images and two text mask images - one for bright
  // text on a dark background, and one for dark text on a bright background

  for (i = 0; i < 2; i++)
  {
    // Select light/dark text

    if ((i == 0) && !config.lightText)
      continue;

    if ((i == 1) && !config.darkText)
      break;

    // Compute SWT

    m_SWT.getTransform(pInputImage,
                       pSWTImage[i],
                       config.cannyLowThreshold,
                       config.cannyHighThreshold,
                       config.angleRangeDivisor,
                       i);

    if (config.separateBlobs)
    {
      // Morphological opening

      cvErode(pSWTImage[i],
              pTempImage,
              NULL,
              config.separateFilterSize);

      cvDilate(pTempImage,
               pSWTImage[i],
               NULL,
               config.separateFilterSize);
    }

    if (config.smoothSWT)
    {
      // Smooth the SWT image

      cvSmooth(pSWTImage[i],
               pTempImage,
               CV_MEDIAN,
               config.smoothFilterSize);

      cvConvertScale(pTempImage,
                     pSWTImage[i],
                     1.0);
    }

    if (config.mergeBlobs)
    {
      // Morphological closing

      cvDilate(pSWTImage[i],
               pTempImage,
               NULL,
               config.mergeFilterSize);

      cvErode(pTempImage,
              pSWTImage[i],
              NULL,
              config.mergeFilterSize);
    }

    // Threshold to get the (foreground) mask image

    cvCmpS(pSWTImage[i], 0, pSWTMaskImage[i], CMP_GT);

    double minVal;
    double maxVal;

    minVal = 0;

    if (config.logSWT)
    {
      // Get natural logarithm of SWT image

      cvLog(pSWTImage[i], pBlobImage[i]);  // = -700 for zero-valued pixels

      // Find min value on the foreground mask

      cvMinMaxLoc(pBlobImage[i],
                  &minVal, &maxVal,
                  NULL, NULL,
                  pSWTMaskImage[i]);

      // We want to subtract this value from the mask area, essentially
      // so that it starts from 0 before the shifting mentioned below

      minVal *= -1;
    }
    else
    {
      cvConvertScale(pSWTImage[i], pBlobImage[i], 1.0);
    }

    // Shift foreground pixel values up and zero the background.
    // This guards against the background becoming part of a blob
    // in the blob detection

    minVal += config.fgMinValue;

    cvZero(pTempImage);

    cvAddS(pBlobImage[i],
           cvScalarAll(minVal),
           pTempImage,
           pSWTMaskImage[i]); 

    cvConvertScale(pTempImage, pBlobImage[i], 1.0);

    if (config.suppressEdges)
    {
      // Suppress edges in the blob image. "Common" edges between strokes
      //  that are close in the input image cause problems

      cvSub(pBlobImage[i], pBlobImage[i], pBlobImage[i], pEdgeImage);
    }

    // Label and count the blobs

    shapeDetect.LabelContinuousBlobs(pBlobImage[i],
                                     pBlobStats,
                                     &numBlobsFound,
                                     config.fgMinValue,
                                     config.strokePixelDiff,
                                     config.maxNumBlobs);

    // printf("num blobs found = %d\n", numBlobsFound);

    // Blob thinning/selection

    for (j = 0; j < numBlobsFound; j++)
      pBlobSelect[j] = true;

    if (config.suppressSmallRegions)
    {
      SuppressSmallRegions(pBlobStats,
                           numBlobsFound,
                           minBoxSize,
                           pBlobSelect);
    }

    if (config.suppressThickStrokes)
    {
      int maxStrokeWidth;

      maxStrokeWidth =
         (int) floor(pSWTImage[i]->height / config.maxStrokeWidthDivisor);

      SuppressThickStrokes(pSWTImage[i],
                           pBlobImage[i],
                           pBlobStats,
                           numBlobsFound,
                           maxStrokeWidth,
                           pBlobSelect);
    }

    ClearBlobs(pBlobImage[i],
               pBlobStats,
               numBlobsFound,
               pBlobSelect);

    // Threshold to get the (foreground) mask image

    cvCmpS(pBlobImage[i], 0, pSWTMaskImage[i], CMP_GT);

    // For each selected blob, create a mask image,
    // and an upright bounding box, and
    // and push them onto the output vectors

    for (j = 0; j < numBlobsFound; j++)
    {
      if (pBlobSelect[j])
      {
        GetBlobMask(pBlobImage[i],
                    pBlobStats,
                    j,
                    blobMask,
                    config.dilateRadius,
                    config.padWidth); 

        textMasks.push_back(blobMask);

        CvRect outputROI;

        PadRegion(pBlobStats[j].rect,
                  imageSize,
                  config.dilateRadius + config.padWidth,
                  outputROI);

        blobRect = Rect(outputROI);
        textBoxes.push_back(blobRect);
      }
    }

    // Prepare display images

    if (config.displayImages)
    {
      Normalize(pSWTImage[i], pSWTDisplayImage[i]);
      Outline(pBlobImage[i], pBlobDisplayImage[i]);
    }
  }

  // Create and save an image with the boxes as an overlay over the blobs

  CreateBoxImage(textBoxes);

  // Display

  if (config.displayImages)
  {
    InitWindows();

    DisplayImages();

    cvDestroyAllWindows();
  }

  // Cleanup

  delete[] pBlobStats;
  delete[] pBlobSelect;

  // Performance stats

  numBlobsFound = textMasks.size();

  // printf("num blobs selected = %d\n", numBlobsFound);
}

void TextDetect::GetImage(int lightDark, char *type, IplImage *pDestImage)
{
  if (strcmp(type, "EDGE") == 0)
    cvCopy(pEdgeImage, pDestImage);
  else if (strcmp(type, "SWT") == 0)
    cvCopy(pSWTImage[lightDark], pDestImage);
  else if (strcmp(type, "BLOB") == 0)
    cvCopy(pBlobImage[lightDark], pDestImage);
  else if (strcmp(type, "MASK") == 0)
    cvCopy(pSWTMaskImage[lightDark], pDestImage);
}

void TextDetect::GetSWTs(Mat *srcImage,
                         Mat *swtImage[2])
{
  int i;
  IplImage srcImageIpl;
  IplImage *pSrcImage;
  IplImage *pMonoImage;
  IplImage *pSWTImage[2];
  CvSize imageSize;
  bool isColor;

  // Input setup

  srcImageIpl = Mat(*srcImage);
  pSrcImage = &srcImageIpl;

  // Create images

  imageSize.width  = pSrcImage->width;
  imageSize.height = pSrcImage->height;

  for (i = 0; i < 2; i++)
    pSWTImage[i] = cvCreateImage(imageSize, IPL_DEPTH_32F, 1);

  // Create mono input image for SWTs

  if (pSrcImage->nChannels == 1)
  {
     isColor = false;
     pMonoImage = pSrcImage;
  }
  else
  {
    isColor = true;
    pMonoImage = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
    cvCvtColor(pSrcImage, pMonoImage, CV_RGB2GRAY);
  }

  // We compute two SWT images and two text mask images - one for bright
  // text on a dark background, and one for dark text on a bright background

  for (i = 0; i < 2; i++)
  {
    // Select light/dark text

    if ((i == 0) && !config.lightText)
      continue;

    if ((i == 1) && !config.darkText)
      break;

    // The transform

    m_SWT.getTransform(pMonoImage,
                       pSWTImage[i],
                       config.cannyLowThreshold,
                       config.cannyHighThreshold,
                       config.angleRangeDivisor,
                       i);
  }

  // Output

  for (i = 0; i < 2; i++)
    swtImage[i] = new Mat(pSWTImage[i], true);

  // Cleanup

  for (i = 0; i < 2; i++)
    cvReleaseImage(&pSWTImage[i]);

  if (isColor)
    cvReleaseImage(&pMonoImage);
}
