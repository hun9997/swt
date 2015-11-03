#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

Size RotatedImageSize(Size inputSize, double angleDegrees)
{
  // Input dimensions

  int inputWidth;
  int inputHeight;

  // Rotation angle

  double angleRadians;
  double alpha;
  double beta;

  // X and Y vectors

  double xLen;
  double yLen;

  Point2f xVec;
  Point2f yVec;

  // Output dimensions

  double outputWidth;
  double outputHeight;
  Size outputSize;

  // Setup

  inputWidth  = inputSize.width;
  inputHeight = inputSize.height;

  // Angle

  angleRadians = (M_PI / 180.0) * angleDegrees;

  alpha = cos(angleRadians);
  beta  = sin(angleRadians);

  // Compute the rotated vectors to the "top" (yVec) and to the "right" (xVec)

  xVec.x = alpha * inputWidth; 
  xVec.y = beta  * inputWidth; 

  yVec.x = beta  * inputHeight; 
  yVec.y = alpha * inputHeight; 

  // Computations for the output dimensions

  outputWidth  = fabs(xVec.x) + fabs(yVec.x);
  outputHeight = fabs(xVec.y) + fabs(yVec.y);

  // Return

  outputSize.width  = (int) ceil(outputWidth + 0.5);
  outputSize.height = (int) ceil(outputHeight + 0.5);

  return outputSize;
}

Mat* RotateImage(Mat &inputImage,
                 double angleDegrees,
                 bool isBinary)
{
  Mat mapMatrix;
  Point2f inputCenter;
  Point2f outputCenter;
  Size inputSize;
  Size rotatedSize;
  Mat *pRotatedImage;
  double tempVal;
  int interpMode;
  int borderMode;

  // Create the Mat to receive the rotated image without cropping

  inputSize = inputImage.size();

  rotatedSize = RotatedImageSize(inputSize, angleDegrees);

  pRotatedImage = new Mat(rotatedSize, CV_8UC1);

  // Image centers

  inputCenter.x = 0.5 * inputSize.width;
  inputCenter.y = 0.5 * inputSize.height;

  outputCenter.x = 0.5 * pRotatedImage->size().width;
  outputCenter.y = 0.5 * pRotatedImage->size().height;

  // Compute the rotation matrix
  //
  // mapMatrix is returned as a 2x3 64-bit-double matrix

  mapMatrix = getRotationMatrix2D(inputCenter, angleDegrees, 1.0);

  // Adjust to center rotated image in output MAT

  tempVal = mapMatrix.at<double>(0, 2);
  tempVal = tempVal - inputCenter.x + outputCenter.x;
  mapMatrix.at<double>(0, 2) = tempVal;

  tempVal = mapMatrix.at<double>(1, 2);
  tempVal = tempVal - inputCenter.y + outputCenter.y;
  mapMatrix.at<double>(1, 2) = tempVal;

  // The rotation itself

  if (isBinary)
  {
    interpMode = INTER_NEAREST;
    borderMode = BORDER_CONSTANT;
  }
  else
  {
    interpMode = INTER_LINEAR;
    borderMode = BORDER_REPLICATE;
  }

  warpAffine(inputImage,
             *pRotatedImage,
             mapMatrix,
             pRotatedImage->size(),
             interpMode,
             borderMode,
             0);

  // Return the output image

  return pRotatedImage;
}

void AlignRegionMat(Mat &inputImage,
                    Mat &outputImage,
                    RotatedRect &minBox,
                    bool isBinary)
{
  Mat mapMatrix;
  Point2f inputCenter;
  Point2f outputCenter;
  Size inputSize;
  double tempVal;
  double angleDegrees;
  int interpMode;
  int borderMode;

  // Image centers

  inputCenter = minBox.center;

  outputCenter.x = 0.5 * outputImage.cols;
  outputCenter.y = 0.5 * outputImage.rows;

  // Compute the rotation matrix

  angleDegrees = minBox.angle;
  mapMatrix = getRotationMatrix2D(inputCenter, angleDegrees, 1.0);

  // Adjust to center rotated image in output Mat

  tempVal = mapMatrix.at<double>(0, 2);
  tempVal = tempVal - inputCenter.x + outputCenter.x;
  mapMatrix.at<double>(0, 2) = tempVal;

  tempVal = mapMatrix.at<double>(1, 2);
  tempVal = tempVal - inputCenter.y + outputCenter.y;
  mapMatrix.at<double>(1, 2) = tempVal;

  // The rotation itself

  if (isBinary)
  {
    interpMode = INTER_NEAREST;
    borderMode = BORDER_CONSTANT;
  }
  else
  {
    interpMode = INTER_LINEAR;
    borderMode = BORDER_REPLICATE;
  }

  warpAffine(inputImage,
             outputImage,
             mapMatrix,
             outputImage.size(),
             interpMode,
             borderMode,
             0);
}

void AlignRegionIpl(IplImage *pInputImage,
                    IplImage *pOutputImage,
                    cv::RotatedRect &minBox)
{
  CvPoint2D32f inputCenter;
  CvPoint2D32f outputCenter;
  CvMat *pMapMatrix;
  float tempVal;
  double angleDegrees;

  // Setup

  inputCenter.x = (int) float(minBox.center.x);
  inputCenter.y = (int) float(minBox.center.y);

  outputCenter.x = 0.5 * pOutputImage->width;
  outputCenter.y = 0.5 * pOutputImage->height;

  angleDegrees = minBox.angle;

  // Output rotation-and-shift matrix

  pMapMatrix = cvCreateMat(2, 3, CV_32F);

  cv2DRotationMatrix(inputCenter, -angleDegrees, 1.0, pMapMatrix);

  // Adjust to center rotated input image in output image

  tempVal = cvmGet(pMapMatrix, 0, 2);
  tempVal = tempVal - inputCenter.x + outputCenter.x;
  cvmSet(pMapMatrix, 0, 2, tempVal);

  tempVal = cvmGet(pMapMatrix, 1, 2);
  tempVal = tempVal - inputCenter.y + outputCenter.y;
  cvmSet(pMapMatrix, 1, 2, tempVal);
  
  // Rotate and shift the input

  // cvWarpAffine(pInputImage, pOutputImage, pMapMatrix, IPL_BORDER_REPLICATE);
  cvWarpAffine(pInputImage, pOutputImage, pMapMatrix);

  // Cleanup

  cvReleaseMat(&pMapMatrix);
}

void AlignPoints(vector<Point2f> &inCorners,
                 vector<Point2f> &outCorners,
                 cv::RotatedRect &minBox)
{
  int i;
  int numInCorners;
  double angle;
  Point2f inCenter;
  Point2f outCenter;
  Point2f tempPoint;
  double x;
  double y;
  double sinAngle;
  double cosAngle;

  // Setup

  outCorners.clear();
  numInCorners = inCorners.size();

  angle    = minBox.angle;
  inCenter = minBox.center;

  outCenter.x = 0.5 * minBox.size.width;
  outCenter.y = 0.5 * minBox.size.height;

  sinAngle = sin(angle);
  cosAngle = cos(angle);

  // Loop through the input points

  for (i = 0; i < numInCorners; i++)
  {
    // Subtract the input center

    x = inCorners[i].x - inCenter.x;
    y = inCorners[i].y - inCenter.y;

    // Rotate by -angle
    // Note that increasing y is downwards, so the rotation
    // matrix looks inverted

    tempPoint.x = cosAngle * x - sinAngle * y;
    tempPoint.y = cosAngle * y + sinAngle * x;

    // Add output center

    tempPoint += outCenter;

    // Push onto output vector

    outCorners.push_back(tempPoint);
  }
}

void AlignPoints2(vector<Point2f> &inPoints,
                 vector<Point2f> &outPoints,
                 Point2f center,
                  double angle)
{
  int i;
  int numInPoints;
  Point2f tempPoint;
  double x;
  double y;
  double sinAngle;
  double cosAngle;

  // Setup

  outPoints.clear();
  numInPoints = inPoints.size();

  sinAngle = sin(angle);
  cosAngle = cos(angle);

  // Loop through the input points

  for (i = 0; i < numInPoints; i++)
  {
    // Subtract the input center

    x = inPoints[i].x - center.x;
    y = inPoints[i].y - center.y;

    // Rotate by -angle
    // Note that increasing y is downwards, so the rotation
    // matrix looks inverted

    tempPoint.x = cosAngle * x - sinAngle * y;
    tempPoint.y = cosAngle * y + sinAngle * x;

    // Push onto output vector

    outPoints.push_back(tempPoint);
  }
}
