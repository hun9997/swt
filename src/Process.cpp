#include "Process.h"

#include <cmath>
#include <iostream>
#include <vector>

#include <time.h>
#include <utility>
#include <algorithm>

using namespace std;
using namespace cv;

// Median-based background subtraction to enhance text and logos

void subtractBG(IplImage *pSrcImage,
                IplImage *pDestImage,
                int windowSize)
{
  IplImage *pTempImage1;
  IplImage *pTempImage2;
  int imageWidth;
  int imageHeight;
  CvSize imageSize;
  double minVal;
  double maxVal;
  double scaleFactor;
  double shift;

  // Setup

  imageWidth  = pSrcImage->width;
  imageHeight = pSrcImage->height;

  imageSize.width  = imageWidth;
  imageSize.height = imageHeight;

  pTempImage1 = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
  pTempImage2 = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);

  // Large median filter - now 2D, but should be separable

  cvSmooth(pSrcImage, pTempImage1, CV_MEDIAN, windowSize);

  // Either absolute difference or difference shifted to be non-negative

#if 1
  // Absolute difference

  cvAbsDiff(pSrcImage, pTempImage1, pTempImage2);

#else
  // Subtract and shift

  cvNot(pTempImage1, pTempImage1);

  cvAddWeighted(pSrcImage, 0.5, pTempImage1, 0.5, 0.0, pTempImage2);
#endif

  // Scale and shift for 0-255 range

  cvMinMaxLoc(pTempImage2, &minVal, &maxVal);

  scaleFactor = 255.0 / (maxVal - minVal);

  shift = - scaleFactor * minVal;

  cvConvertScale(pTempImage2, pDestImage, scaleFactor, shift);

  // Cleanup

  cvReleaseImage(&pTempImage1);
  cvReleaseImage(&pTempImage2);
}

// Process the SWT to estimate a binary mask image of text pixels

void getTextMask(IplImage *pSWTImage,
                 IplImage *pMaskImage)
{
  int i;
  IplImage *pThreshImage1;;
  IplImage *pThreshImage2;
  IplImage *pTempImage3;
  IplImage *pTempImage4;
  int imageWidth;
  int imageHeight;
  CvSize imageSize;
  IplConvKernel *pMorphKernel;
  int kernelValues[9];

  double lowThreshold;
  double highThreshold;
  double smoothThreshold;

  // Setup

  // Images

  imageWidth  = pSWTImage->width;
  imageHeight = pSWTImage->height;

  imageSize.width  = imageWidth;
  imageSize.height = imageHeight;

  pThreshImage1 = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
  pThreshImage2 = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
  pTempImage3 = cvCreateImage(imageSize, IPL_DEPTH_32F, 1);
  pTempImage4 = cvCreateImage(imageSize, IPL_DEPTH_32F, 1);

  // Morphological kernel

  for (i = 0; i < 9; i ++)
    kernelValues[i] = 1;

  kernelValues[4] = 0;

  pMorphKernel = cvCreateStructuringElementEx(3, 3, 1, 1,
                                              CV_SHAPE_CUSTOM,
                                              kernelValues);

  // Parameters

  lowThreshold = 0;
  highThreshold = 0.025 * imageWidth;
  // highThreshold = imageWidth;
  // smoothThreshold = 1.0;
  smoothThreshold = 0.5;

  // Find where the SWT is between lowThreshold and highThreshold
  // This is the range of stroke widths for text

  cvCmpS(pSWTImage, lowThreshold,  pThreshImage1, CV_CMP_GT);
  cvCmpS(pSWTImage, highThreshold, pThreshImage2, CV_CMP_LT);

  cvMin(pThreshImage1, pThreshImage2, pMaskImage);

  // SWT smoothness estimate
  // Find pixels that are within a range of their brighter or darker neighbors

  cvDilate(pSWTImage, pTempImage3, pMorphKernel, 1);  // 3x3 max filter without central pixel
  cvErode(pSWTImage,  pTempImage4, pMorphKernel, 1);  // 3x3 min filter without central pixel

  cvAbsDiff(pTempImage3, pSWTImage, pTempImage3);   // abs(max - orig)
  cvCmpS(pTempImage3, smoothThreshold, pThreshImage1, CV_CMP_LE);  // threshold

  cvAbsDiff(pTempImage4, pSWTImage, pTempImage4);   // abs(min - orig)
  cvCmpS(pTempImage4, smoothThreshold, pThreshImage2, CV_CMP_LE);  // threshold

  cvOr(pThreshImage1, pThreshImage2, pThreshImage1);

  cvMin(pThreshImage1, pMaskImage, pMaskImage);  // Combine with the previous condition

  // Fill in gaps in the mask

  // Cleanup

  cvDilate(pMaskImage,   pThreshImage1, NULL, 3);  // was 1 iteration
  cvErode(pThreshImage1, pMaskImage,    NULL, 3);

  cvReleaseImage(&pThreshImage1);
  cvReleaseImage(&pThreshImage2);
  cvReleaseImage(&pTempImage3);
  cvReleaseImage(&pTempImage4);

  cvReleaseStructuringElement(&pMorphKernel);
}

void FindBoxes(IplImage *pLabelledBlobImage,
               vector<Rect> &blobBoxes)
{
  int i;
  int j;
  Mat inputImage;
  Mat paddedImage;
  Mat absDiffImage;
  Mat blobMask;
  int numBlobs;
  double minVal;
  double maxVal;
  double blobLabel;
  vector<vector<Point > > contours;
  Rect boundingBox;
  Rect maxBox;

  blobBoxes.clear();

  inputImage   = Mat(pLabelledBlobImage, true);
  paddedImage  = Mat(inputImage.rows + 2, inputImage.cols + 2, CV_32F);
  absDiffImage = Mat(inputImage.rows + 2, inputImage.cols + 2, CV_32F);
  blobMask     = Mat(inputImage.rows + 2, inputImage.cols + 2, CV_8U);

  // We have to pad the input image with single rows and columns of zeros.
  // This is because contours stop at the boundary of the original image.
  // We need the padding to find the contours we would expect for the blobs.

  copyMakeBorder(inputImage,
                 paddedImage,
                 1, 1, 1, 1,
                 BORDER_CONSTANT,
                 Scalar(0, 0, 0));

  // Find the number of blobs in the labelled image

  minMaxLoc(inputImage, &minVal, &maxVal);

  numBlobs = (int) floor(maxVal);

  // Main loop

  for (i = 0; i < numBlobs; i++)
  {
    // Find the mask of the current labelled pixels

    blobLabel = i + 1;

    // Compute absolute diff from blob label

    absdiff(paddedImage,
            Scalar(blobLabel, 0, 0),
            absDiffImage);

    // Compare to 0.5 threshold to get mask image

    compare(absDiffImage,
            Scalar(0.5, 0, 0),
            blobMask,
            CMP_LT);

    // Find the external contours around the current mask
    // As the mask may consist of multiple blobs, there
    // can be multiple contours.

    contours.clear();
    findContours(blobMask,
                 contours,
                 CV_RETR_EXTERNAL,
                 CV_CHAIN_APPROX_NONE);

    if (contours.size() > 0)
    {
      maxBox = boundingRect(contours[0]);

      for (j = 1; j < contours.size(); j++)
      {
        boundingBox = boundingRect(contours[j]);
        maxBox = maxBox | boundingBox;
      }

      // Subtract unit offsets to adjust for initial padding

      maxBox.x -= 1;
      maxBox.y -= 1;

      // Push onto vector

      blobBoxes.push_back(maxBox);
    }
    else
    {
      printf("FindBoxes: bounding box not found, blob %d\n", i);

      int count;

      count = countNonZero(blobMask);

      printf("FindBoxes: mask area = %d\n", count);
    }
  }
}

void FindBoxesAndMasks(IplImage *pLabelledBlobImage,
                       std::vector<cv::Rect> &blobBoxes,
                       std::vector<cv::Mat> &blobMasks)
{
  int i;
  int j;
  Mat inputImage;
  Mat paddedImage;
  Mat absDiffImage;
  Mat blobImage;
  Mat blobMask;
  int numBlobs;
  double minVal;
  double maxVal;
  double blobLabel;
  vector<vector<Point > > contours;
  Rect boundingBox;
  Rect maxBox;

  blobBoxes.clear();
  blobMasks.clear();

  inputImage   = Mat(pLabelledBlobImage, true);
  paddedImage  = Mat(inputImage.rows + 2, inputImage.cols + 2, CV_32F);
  absDiffImage = Mat(inputImage.rows + 2, inputImage.cols + 2, CV_32F);
  blobImage    = Mat(inputImage.rows + 2, inputImage.cols + 2, CV_8U);

  // We have to pad the input image with single rows and columns of zeros.
  // This is because contours stop at the boundary of the original image.
  // We need the padding to find the contours we would expect for the blobs.

  copyMakeBorder(inputImage,
                 paddedImage,
                 1, 1, 1, 1,
                 BORDER_CONSTANT,
                 Scalar(0, 0, 0));

  // Find the number of blobs in the labelled image

  minMaxLoc(inputImage, &minVal, &maxVal);

  numBlobs = (int) floor(maxVal);

  // Main loop

  for (i = 0; i < numBlobs; i++)
  {
    // Find the mask of the current labelled pixels

    blobLabel = i + 1;

    // Compute absolute diff from blob label

    absdiff(paddedImage,
            Scalar(blobLabel, 0, 0),
            absDiffImage);

    // Compare to 0.5 threshold to get mask image

    compare(absDiffImage,
            Scalar(0.5, 0, 0),
            blobImage,
            CMP_LT);

    // Find the external contours around the current mask
    // As the mask may consist of multiple blobs, there
    // can be multiple contours.

    contours.clear();
    findContours(blobImage,
                 contours,
                 CV_RETR_EXTERNAL,
                 CV_CHAIN_APPROX_NONE);

    if (contours.size() > 0)
    {
      maxBox = boundingRect(contours[0]);

      for (j = 1; j < contours.size(); j++)
      {
        boundingBox = boundingRect(contours[j]);
        maxBox = maxBox | boundingBox;
      }

      // Get the mask

      blobMask = Mat(blobImage, maxBox);

      // Subtract unit offsets to adjust for initial padding

      maxBox.x -= 1;
      maxBox.y -= 1;

      // Push onto output vectors

      blobMasks.push_back(blobMask);
      blobBoxes.push_back(maxBox);
    }
    else
    {
      printf("FindBoxes: bounding box not found, blob %d\n", i);

      int count;

      count = countNonZero(blobMask);

      printf("FindBoxes: mask area = %d\n", count);
    }
  }
}

void DrawBoxes(IplImage *pDisplayImage,
               vector<Rect> blobBoxes)
{
  int i;

  for (i = 0; i < blobBoxes.size(); i++)
  {
    DrawBox(pDisplayImage, blobBoxes[i]);
  }
}

void DrawBox(IplImage *pDisplayImage,
             Rect &blobBox)
{
  int i;
  CvPoint corner[4];
  CvScalar color;

  color = cvScalar(0, 0, 255, 0);

  corner[0].x = 0;
  corner[0].y = 0;

  corner[1].x = blobBox.width;
  corner[1].y = 0;

  corner[2].x = blobBox.width;
  corner[2].y = blobBox.height;

  corner[3].x = 0;
  corner[3].y = blobBox.height;

  for (i = 0; i < 4; i++)
  {
    corner[i].x += blobBox.x;
    corner[i].y += blobBox.y;
  }

  for (i = 0; i < 4; i++)
    cvLine(pDisplayImage, corner[i], corner[(i + 1) % 4], color);
}

void FindMinBoxes(vector<Mat> &blobMasks,
                  vector<Rect> &blobBoxes,
                  vector<RotatedRect> &blobMinBoxes)
{
  int i;
  int j;
  int k;
  Mat inputMask;
  int numBlobs;
  vector<vector<Point > > contours;
  vector<Point> allPoints;
  RotatedRect minBox;

  numBlobs = blobMasks.size();

  blobMinBoxes.clear();

  for (i = 0; i < numBlobs; i++)
  {
    contours.clear();
    inputMask = Mat(blobMasks[i].rows + 2, blobMasks[i].cols + 2, CV_8U);

    copyMakeBorder(blobMasks[i],
                   inputMask,
                   1, 1, 1, 1,
                   BORDER_CONSTANT,
                   Scalar(0, 0, 0));

    // Find the contours around the current mask

    findContours(inputMask,
                 contours,
                 CV_RETR_EXTERNAL,
                 CV_CHAIN_APPROX_NONE);

    // Merge the contours into one vector

    allPoints.clear();

    for (j = 0; j < contours.size(); j++)
    {
      for (k = 0; k < contours[j].size(); k++)
      {
        allPoints.push_back((contours[j])[k]);
      }
    }

    // Find the rectangle

    if (allPoints.size() > 0)
    {
      // Find the minimum bounding box for contour[0]

      minBox = minAreaRect(Mat(allPoints));

      // Add the offset for the position in the source image

      minBox.center.x += blobBoxes[i].x - 1;
      minBox.center.y += blobBoxes[i].y - 1;
    }
    else
    {
      // Includes offset of center

      minBox.angle  = 0.0;
      minBox.size.width  = blobBoxes[i].width;
      minBox.size.height = blobBoxes[i].height;
      minBox.center.x = blobBoxes[i].x - 0.5 * minBox.size.width;
      minBox.center.y = blobBoxes[i].y - 0.5 * minBox.size.height;
    }

    // Push onto vector

    blobMinBoxes.push_back(minBox);
  }

}

void DrawMinBoxes(IplImage *pDisplayImage,
                  vector<RotatedRect> &blobBoxes,
                  int special)
{
  int i;

  for (i = 0; i < blobBoxes.size(); i++)
    DrawMinBox(pDisplayImage, blobBoxes[i], special);
}

void DrawMinBox(IplImage *pDisplayImage,
                RotatedRect &blobBox,
                int special)
{
  int i;
  Point2f vertices[4];
  CvPoint corner[4];
  CvScalar color;

  blobBox.points(vertices);

  if (special == 0)
    color = cvScalar(0, 0, 255, 0);
  else
    color = cvScalar(0, 255, 0, 0);

  for (i = 0; i < 4; i++)
  {
    corner[i].x = (int) floor(vertices[i].x + 0.5);
    corner[i].y = (int) floor(vertices[i].y + 0.5);
  }

  for (i = 0; i < 4; i++)
    cvLine(pDisplayImage, corner[i], corner[(i + 1) % 4], color);
}

void GetFlatRegions(IplImage *pOrigImage,
                    IplImage *pLabelledBlobs,
                    CvConnectedComp *pBlobStats,
                    int numBlobs,
                    float flatThreshold,
                    IplImage *pOutputMask)
{
  int blobIndex;
  IplImage *pMaskImage;
  CvSize imageSize;
  CvRect roi;
  float blobLabel;
  CvScalar meanVec;
  CvScalar stdVec;
  float mean;
  float stdDev;
  bool isFlat;

  imageSize.width  = pOrigImage->width;
  imageSize.height = pOrigImage->height;

  pMaskImage = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);

  cvZero(pMaskImage);
  cvZero(pOutputMask);

  for (blobIndex = 0; blobIndex < numBlobs; blobIndex++)
  {
#if 0
    // Set ROI on all images

    roi = pBlobStats[blobIndex].rect;

    cvSetImageROI(pOrigImage, roi);
    cvSetImageROI(pLabelledBlobs, roi);
    cvSetImageROI(pMaskImage, roi);
    cvSetImageROI(pOutputMask, roi);
#endif

    // Set mask for current blob

    blobLabel = blobIndex + 1;

    cvCmpS(pLabelledBlobs,
           blobLabel,
           pMaskImage,
           CV_CMP_EQ);

    // Find average and standard deviation in the blob

    cvAvgSdv(pOrigImage, &meanVec, &stdVec, pMaskImage);

    mean   = meanVec.val[0];
    stdDev = stdVec.val[0];

    // printf("index = %d mean = %f stddev = %f\n", blobIndex, mean, stdDev);

    // Flatness decision

    if (stdDev < flatThreshold)
      isFlat = true;
    else
      isFlat = false;

    // If flat, we add the blob to the output mask

    if (isFlat)
      cvOr(pMaskImage, pOutputMask, pOutputMask);

    // Set the internal mask image to 0 in the ROI

    cvZero(pMaskImage);
    
  }

  // Reset all image ROIs

  cvResetImageROI(pOrigImage);
  cvResetImageROI(pLabelledBlobs);
  cvResetImageROI(pMaskImage);
  cvResetImageROI(pOutputMask);

  cvReleaseImage(&pMaskImage);
}

// Reset select to false for all small blobs

void SuppressSmallRegions(CvConnectedComp *pBlobStats,
                          int numBlobs,
                          int minSize,
                          bool *pBlobSelect)
{
  CvRect roi;
  int blobIndex;
  bool smallBlob;

  for (blobIndex = 0; blobIndex < numBlobs; blobIndex++)
  {
    roi = pBlobStats[blobIndex].rect;

    if ((roi.width < minSize) && (roi.height < minSize))
      pBlobSelect[blobIndex] = false;
  }
}

// Reset select to false for all large blobs

void SuppressLargeRegions(CvConnectedComp *pBlobStats,
                          int numBlobs,
                          int maxSize,
                          bool *pBlobSelect)
{
  CvRect roi;
  int blobIndex;
  bool largeBlob;

  for (blobIndex = 0; blobIndex < numBlobs; blobIndex++)
  {
    roi = pBlobStats[blobIndex].rect;

    if ((roi.width > maxSize) && (roi.height > maxSize))    // changed from && = and
      pBlobSelect[blobIndex] = false;
  }
}

// Pad a subimage area

void PadRegion(CvRect &inputROI,
               CvSize &imageSize,
               int padWidth,
               CvRect &outputROI)
{
  int delta;

  // Initial padding

  outputROI.width  = inputROI.width  + 2 * padWidth;
  outputROI.height = inputROI.height + 2 * padWidth;

  outputROI.x = inputROI.x - padWidth;
  outputROI.y = inputROI.y - padWidth;

  // Clipping to image area

  // Left

  delta = outputROI.x;

  if (delta < 0)
  {
    outputROI.x = 0;
    outputROI.width += delta;
  }

  // Right

  delta = outputROI.x + outputROI.width - imageSize.width;

  if (delta > 0)
    outputROI.width -= delta;

  // Top

  delta = outputROI.y;

  if (delta < 0)
  {
    outputROI.y = 0;
    outputROI.height += delta;
  }

  // Bottom

  delta = outputROI.y + outputROI.height - imageSize.height;

  if (delta > 0)
    outputROI.height -= delta;
}

// Extract shape of a blob within its bounding box

void GetBlobMask(IplImage *pBlobImage,
                 CvConnectedComp *pBlobStats,
                 int blobIndex,
                 Mat &blobMask,
                 int dilateRadius,
                 int borderWidth)
{
  CvRect roi;
  CvSize imageSize;
  CvSize maskSize;
  int padWidth;
  IplImage *pMaskImage;
  IplImage *pTempImage;

  // Get the ROI

  imageSize.width  = pBlobImage->width;
  imageSize.height = pBlobImage->height;

  padWidth = dilateRadius + borderWidth;

  PadRegion(pBlobStats[blobIndex].rect,
            imageSize,
            padWidth,
            roi);

  // Create the output image

  maskSize.width  = roi.width;
  maskSize.height = roi.height;

  pMaskImage = cvCreateImage(maskSize, IPL_DEPTH_8U, 1);
  pTempImage = cvCreateImage(maskSize, IPL_DEPTH_8U, 1);

  // Set ROI to known bounding box to speed computation

  cvSetImageROI(pBlobImage, roi);

  // Get the mask

  cvCmpS(pBlobImage, blobIndex + 1, pMaskImage, CMP_EQ);

  // Dilate it

  if (dilateRadius > 0)
  {
    cvDilate(pMaskImage, pTempImage, NULL, dilateRadius);
    cvCopy(pTempImage, pMaskImage);
  }

  // Convert it to a Mat for openCV 2.x+

  blobMask = Mat(pMaskImage, true);

  // Clean up

  cvResetImageROI(pBlobImage);
  cvReleaseImage(&pMaskImage);
  cvReleaseImage(&pTempImage);
}

// Extract shape of a blob within its bounding box (second version)

void GetBlobMask2(IplImage *pBlobImage,
                  int blobIndex,
                  Rect &blobBox,
                  Mat &blobMask,
                  int dilateRadius,
                  int borderWidth)
{
  CvRect roi;
  CvSize imageSize;
  CvSize maskSize;
  int padWidth;
  IplImage *pMaskImage;
  IplImage *pTempImage;
  CvRect rect;

  // Get the ROI

  imageSize.width  = pBlobImage->width;
  imageSize.height = pBlobImage->height;

  padWidth = dilateRadius + borderWidth;

  rect.x = blobBox.x;
  rect.y = blobBox.y;
  rect.width = blobBox.width;
  rect.height = blobBox.height;

  PadRegion(rect,
            imageSize,
            padWidth,
            roi);

  // Create the output image

  maskSize.width  = roi.width;
  maskSize.height = roi.height;

  pMaskImage = cvCreateImage(maskSize, IPL_DEPTH_8U, 1);
  pTempImage = cvCreateImage(maskSize, IPL_DEPTH_8U, 1);

  // Set ROI to known bounding box to speed computation

  cvSetImageROI(pBlobImage, roi);

  // Get the mask

  cvCmpS(pBlobImage, blobIndex + 1, pMaskImage, CMP_EQ);

  // Dilate it

  if (dilateRadius > 0)
  {
    cvDilate(pMaskImage, pTempImage, NULL, dilateRadius);
    cvCopy(pTempImage, pMaskImage);
  }

  // Convert it to a Mat for openCV 2.x+

  blobMask = Mat(pMaskImage, true);

  // Clean up

  cvResetImageROI(pBlobImage);
  cvReleaseImage(&pMaskImage);
  cvReleaseImage(&pTempImage);
}

void GetBlobMasks(IplImage *pBlobImage,
                  vector<cv::Rect> &blobBoxes,
                  vector<cv::Mat> &blobMasks)
{
  int numBlobs;
  int i;
  Mat blobMask;

  numBlobs = blobBoxes.size();
  blobMasks.clear();
 
  for (i = 0; i < numBlobs; i++)
  {
    GetBlobMask2(pBlobImage,
                 i,
                 blobBoxes[i],
                 blobMask);

    blobMasks.push_back(blobMask);
  }
}

// Find the maximum stroke width in a blob

float GetMaxStrokeWidth(IplImage *pSWTImage,
                        IplImage *pBlobImage,
                        CvConnectedComp *pBlobStats,
                        int blobIndex)
{
  CvRect roi;
  CvSize imageSize;
  IplImage *pMaskImage;
  float blobValue;
  double minValue;
  double maxValue;

  // Init

  roi = pBlobStats[blobIndex].rect;

  imageSize.width  = pBlobImage->width;
  imageSize.height = pBlobImage->height;

  blobValue = blobIndex + 1;

  // Create the mask image

  pMaskImage = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);

  // Set ROI to known bounding box to speed computation

  cvSetImageROI(pSWTImage,  roi);
  cvSetImageROI(pBlobImage, roi);
  cvSetImageROI(pMaskImage, roi);

  // Find the blob

  cvCmpS(pBlobImage, blobValue, pMaskImage, CMP_EQ);

  // Find the max and min stroke widths

  cvMinMaxLoc(pSWTImage, &minValue, &maxValue, NULL, NULL, pMaskImage);

  // Clean up

  cvResetImageROI(pSWTImage);
  cvResetImageROI(pBlobImage);
  cvResetImageROI(pMaskImage);
  cvReleaseImage(&pMaskImage);

  // Return max stroke width in the blob

  return (float) maxValue;
}

// Set select to false for all blobs with high maximum stroke width
// We skip all blobs that are already deselected

void SuppressThickStrokes(IplImage *pSWTImage,
                          IplImage *pBlobImage,
                          CvConnectedComp *pBlobStats,
                          int numBlobs,
                          int maxStrokeWidth,
                          bool *pBlobSelect)
{
  int i;
  float strokeWidth;

  for (i = 0; i < numBlobs; i++)
  {
     if (pBlobSelect[i] == false)
       continue;

    // printf("large region index = %d\n", i);

    strokeWidth = GetMaxStrokeWidth(pSWTImage,
                                    pBlobImage,
                                    pBlobStats,
                                    i);

    if (strokeWidth > (float) maxStrokeWidth) 
      pBlobSelect[i] = false;
  }
}

// Clear the pixels for a blob

void ClearBlob(IplImage *pBlobImage,
               CvConnectedComp *pBlobStats,
               int blobIndex)
{
  CvRect roi;
  CvSize imageSize;
  IplImage *pMaskImage;
  float blobValue;

  // Init

  roi = pBlobStats[blobIndex].rect;

  imageSize.width  = pBlobImage->width;
  imageSize.height = pBlobImage->height;

  blobValue = blobIndex + 1;

  // Create the mask image

  pMaskImage = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);

  // Set ROI to known bounding box to speed computation

  cvSetImageROI(pBlobImage, roi);
  cvSetImageROI(pMaskImage, roi);

  // Find the blob pixels

  cvCmpS(pBlobImage, blobValue, pMaskImage, CMP_EQ);

  // Clear the blob pixels

  cvSubS(pBlobImage, 
         cvScalarAll(blobValue),
         pBlobImage,
         pMaskImage);

  // Clean up

  cvResetImageROI(pBlobImage);
  cvResetImageROI(pMaskImage);
  cvReleaseImage(&pMaskImage);
}

// Clear pixels for all de-selected blobs

void ClearBlobs(IplImage *pBlobImage,
                CvConnectedComp *pBlobStats,
                int numBlobs,
                bool *pBlobSelect)
{
  int i;

  for (i = 0; i < numBlobs; i++)
  {
    if (pBlobSelect[i] == false)
    {
      // printf("clear blob index = %d\n", i);
      ClearBlob(pBlobImage, pBlobStats, i);
    }
  }
}

void Normalize(IplImage *pSrcImage,
               IplImage *pDestImage)
{
  double maxValue;
  double minValue;
  double shift;
  double scaleFactor;

  cvMinMaxLoc(pSrcImage, &minValue, &maxValue, NULL, NULL, NULL);

  // printf("minValue = %f  maxValue = %f\n", minValue, maxValue);

  if (minValue != maxValue)
  {
    scaleFactor = 255.0 / (maxValue - minValue); 

    shift = -minValue / scaleFactor;
  }
  else
  {
    scaleFactor = 1.0;
    shift = 0;
  }

  cvConvertScale(pSrcImage, pDestImage, scaleFactor, shift);
}

void Outline(IplImage *pSrcImage,
             IplImage *pDestImage)
{
  int imageHeight;
  int imageWidth;
  CvSize imageSize;
  IplImage *pTempImage;

  // Setup

  // Images

  imageWidth  = pSrcImage->width;
  imageHeight = pSrcImage->height;

  imageSize.width  = imageWidth;
  imageSize.height = imageHeight;

  pTempImage = cvCreateImage(imageSize, pSrcImage->depth, 1);

  // Processing

  cvDilate(pSrcImage, pTempImage, NULL, 3);
  cvAbsDiff(pTempImage, pSrcImage, pTempImage);
  cvCmpS(pTempImage, 1.0, pDestImage, CV_CMP_GE);

  // Cleanup

  cvReleaseImage(&pTempImage);
}
