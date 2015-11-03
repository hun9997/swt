#include "ConvertUtils.h"

using namespace std;
using namespace cv;
using namespace gnim;
using namespace papyrus;

// Convert IplImage to gnim::Image

void IplToGNIM(IplImage *pSrcImage,
              Image **pDestImage)
{
  int numChannels;
  int pixelSize;
  int iplType;
  unsigned int iplDepth;
  gnim_pixel_type_t gnimType;
  int pixelDepth;

  int width;
  int height;
  int row;
  int col;

  int rowSize;

  unsigned char *pSrcData;
  unsigned char *pSrcRow;
  int SrcStep;
  CvSize SrcSize;

  unsigned char *pDestData;
  unsigned char *pDestRow;
  int DestStep;

  Image *pGNIMImage;

  // Setup

  width  = pSrcImage->width;
  height = pSrcImage->height;
  numChannels = pSrcImage->nChannels;
  iplDepth    = pSrcImage->depth;

  // Pixel depth

  switch(iplDepth)
  {
    case IPL_DEPTH_8U:
    case IPL_DEPTH_8S:
      pixelDepth = 1;
      break;
    default:
      printf("IplToGNIM: pixel depth not supported\n");
      *pDestImage = NULL;
      return;
  }

  // Number of channels  => gnim pixel type

  switch(numChannels)
  {
    case 1:
      gnimType = GNIM_PIX_GRAYSCALE8;
      break;
    case 3:
      gnimType = GNIM_PIX_BGR24;
      break;
    case 4:
      gnimType = GNIM_PIX_ARGB32;
      break;
    default:
      printf("IplToGNIM: number of channels not supported\n");
      *pDestImage = NULL;
      return;
  }

  // Get access to the IplImage data

  cvGetRawData(pSrcImage,
               (uchar**)&pSrcData,
               &SrcStep,
               &SrcSize);

  // Create the gnim Image

  pGNIMImage = new Image(height, width, gnimType, NULL, 0);

  // Get access to the CCV data

  pDestData = pGNIMImage->buffer();
  DestStep  = pGNIMImage->row_stride();

  // Copy the pixel data

  pSrcRow  = pSrcData;
  pDestRow = pDestData;

  pixelSize = numChannels * pixelDepth;
  rowSize   = pixelSize   * width;

  for (row = 0; row < height; row++)
  {
    // Copy one row

    memcpy(pDestRow, pSrcRow, rowSize);

    // Increment to the next row
    // Note that the steps are in bytes, not pixels

    pSrcRow  += SrcStep;
    pDestRow += DestStep;
  }

  // Return

  *pDestImage = pGNIMImage;
}

// Convert gnim::Image to IplImage

void GNIMToIpl(const Image *pSrcImage,
               IplImage **pDestImage)
{
  int numChannels;
  int pixelSize;
  gnim_pixel_type_t gnimType;
  int gnimDepth;
  int iplDepth;

  int width;
  int height;
  int row;
  int col;

  int rowSize;

  unsigned char *pSrcData;
  unsigned char *pSrcRow;
  int SrcStep;

  unsigned char *pDestData;
  unsigned char *pDestRow;
  int DestStep;
  CvSize DestSize;

  IplImage *pIplImage;

  // Setup

  width  = pSrcImage->width();
  height = pSrcImage->height();
  gnimDepth = pSrcImage->depth();
  gnimType  = pSrcImage->pixel_type();

  // Pixel depth always 8 bits per component;
  // Number of channels = GNIM "depth"

  iplDepth = IPL_DEPTH_8U;
  numChannels = gnimDepth;
  pixelSize = numChannels;

  // Create the IplImage

  DestSize.width  = width;
  DestSize.height = height;

  pIplImage = cvCreateImage(DestSize, iplDepth, numChannels);

  // Get access to the GNIM data

  pSrcData = (unsigned char *) pSrcImage->buffer();
  SrcStep  = pSrcImage->row_stride();

  // Get access to the IplImage data

  cvGetRawData(pIplImage,
               (uchar**)&pDestData,
               &DestStep,
               &DestSize);

  // Copy the pixel data

  pSrcRow  = pSrcData;
  pDestRow = pDestData;

  rowSize = pixelSize * width;

  for (row = 0; row < height; row++)
  {
    // Copy one row

    memcpy(pDestRow, pSrcRow, rowSize);

    // Increment to the next row
    // Note that the steps are in bytes, not pixels

    pSrcRow  += SrcStep;
    pDestRow += DestStep;
  }

  // Return

  *pDestImage = pIplImage;
}

// Invert the pixel values in an 8-bit mono GNIM image

void InvertGNIM(const gnim::Image *pSrcImage,
                gnim::Image **pDestImage)
{
  int numChannels;
  int pixelSize;
  gnim_pixel_type_t gnimType;
  int gnimDepth;

  int width;
  int height;
  int row;
  int col;

  unsigned char *pSrcData;
  unsigned char *pSrcRow;
  unsigned char *pSrcPix;
  int SrcStep;

  unsigned char *pDestData;
  unsigned char *pDestRow;
  unsigned char *pDestPix;
  int DestStep;

  gnim::Image *pOutImage;

  // Setup

  width  = pSrcImage->width();
  height = pSrcImage->height();
  gnimDepth = pSrcImage->depth();
  gnimType  = pSrcImage->pixel_type();

  // Pixel depth always 8 bits per component;
  // Number of channels = GNIM "depth"

  numChannels = gnimDepth;

  if (numChannels != 1)
  {
    printf("InvertGNIM: only mono images supported\n");
    return;
  }

  // Create the output image

  pOutImage = new Image(height, width, gnimType, NULL, 0);

  // Get access to the GNIM data

  pSrcData  = (unsigned char *) pSrcImage->buffer();
  SrcStep   = pSrcImage->row_stride();

  pDestData = (unsigned char *) pOutImage->buffer();
  DestStep  = pOutImage->row_stride();

  // Invert and copy the pixel data

  pSrcRow  = pSrcData;
  pDestRow = pDestData;

  for (row = 0; row < height; row++)
  {
    // Copy one row

    pSrcPix = pSrcRow;
    pDestPix = pDestRow;

    for (col = 0; col < width; col++)
    {
      *pDestPix = ~(*pSrcPix);

      pSrcPix++;
      pDestPix++;
    }

    // Increment to the next row
    // Note that the steps are in bytes, not pixels

    pSrcRow  += SrcStep;
    pDestRow += DestStep;
  }

  // Return

  *pDestImage = pOutImage;
}

// Convert OpenCV Rect to bounding_box_t

void RectToBox(Rect &rect, bounding_box_t &box)
{
  box.top_left.x = rect.x;
  box.top_left.y = rect.y;

  box.top_right.x = rect.x + rect.width;
  box.top_right.y = rect.y;

  box.bottom_left.x = rect.x;
  box.bottom_left.y = rect.y + rect.height;

  box.bottom_right.x = rect.x + rect.width;
  box.bottom_right.y = rect.y + rect.height;
}

// Convert bounding_box_t to OpenCV Rect

void BoxToRect(bounding_box_t &box, Rect &rect)
{
  int i;

  float xMin;
  float xMax;
  float yMin;
  float yMax;

  // Setup

  float x[4];
  float y[4];

  x[0] = box.top_left.x;
  x[1] = box.top_right.x;
  x[2] = box.bottom_left.x;
  x[3] = box.bottom_right.x;

  y[0] = box.top_left.y;
  y[1] = box.top_right.y;
  y[2] = box.bottom_left.y;
  y[3] = box.bottom_right.y;

  // x limits

  xMin = x[0];

  for (i = 1; i < 4; i++)
    xMin = min(xMin, x[i]);

  xMax = x[0];

  for (i = 1; i < 4; i++)
    xMax = max(xMax, x[i]);

  // y limits

  yMin = y[0];

  for (i = 1; i < 4; i++)
    yMin = min(yMin, y[i]);

  yMax = y[0];

  for (i = 1; i < 4; i++)
    yMax = max(yMax, y[i]);

  // Final computations

  rect.x = xMin;
  rect.y = yMin;

  rect.width  = xMax - xMin;
  rect.height = yMax - yMin;
}

// Convert OpenCV RotatedRect to bounding_box_t

void RotatedRectToBox(RotatedRect &rect, bounding_box_t &box)
{
  Point2f vertices[4];

  rect.points(vertices);

  // Check that the output order is correct

  box.top_left.x     = vertices[1].x;
  box.top_left.y     = vertices[1].y;

  box.top_right.x    = vertices[2].x;
  box.top_right.y    = vertices[2].y;

  box.bottom_right.x = vertices[3].x;
  box.bottom_right.y = vertices[3].y;

  box.bottom_left.x  = vertices[0].x;
  box.bottom_left.y  = vertices[0].y;

#if 0
  // Print results

  cout << "top left x = " << box.top_left.x << " y = " << box.top_left.y << endl;
  cout << "top right x = " << box.top_right.x << " y = " << box.top_right.y << endl;
  cout << "bottom right x = " << box.bottom_right.x << " y = " << box.bottom_right.y << endl;
  cout << "bottom left x = " << box.bottom_left.x << " y = " << box.bottom_left.y << endl;
#endif

}

// Convert bounding_box_t to OpenCV RotatedRect
#if 0
void BoxToRotatedRect(bounding_box_t &box, RotatedRect &rect)
{
  Point2f xVec;
  Point2f yVec;
  Point2f center;

  double width;
  double height;
  double angle;

  xVec.x = box.top_right.x - box.top_left.x;
  xVec.y = box.top_right.y - box.top_left.y;

  yVec.x = box.bottom_left.x - box.top_left.x;
  yVec.y = box.bottom_left.y - box.top_left.y;

  width  = sqrt(xVec.x * xVec.x + xVec.y * xVec.y);
  height = sqrt(yVec.x * yVec.x + yVec.y * yVec.y);

  angle = atan2(xVec.y, xVec.x);

  center.x = box.top_left.x;
  center.y = box.top_left.y;

  center += 0.5 * xVec + 0.5 * yVec;

  rect.center = center;
  rect.angle  = angle;
  rect.size.width  = width;
  rect.size.height = height;
}
#else

// This version just uses minAreaRect()
// This is an experiment because of the curious behavior of
// minAreaRect
// There will be some error because the input points will be
// quantized

void BoxToRotatedRect(bounding_box_t &box, RotatedRect &rect)
{
  vector<Point> vertices; 
  Point vertex;

  // Convert the box to the vertices

  vertex.x = box.bottom_left.x;
  vertex.y = box.bottom_left.y;

  vertices.push_back(vertex);

  vertex.x = box.top_left.x;
  vertex.y = box.top_left.y;

  vertices.push_back(vertex);

  vertex.x = box.top_right.x;
  vertex.y = box.top_right.y;

  vertices.push_back(vertex);

  vertex.x = box.bottom_right.x;
  vertex.y = box.bottom_right.y;

  vertices.push_back(vertex);

  // Compute the RotatedRect

  rect = minAreaRect(Mat(vertices));
}
#endif
