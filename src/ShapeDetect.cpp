//
// ShapeDetect.cpp
//

#include "ShapeDetect.hpp"

ShapeDetect::ShapeDetect()
{
  m_DisplayEnable = true;
  m_FirstFrame  = true;
  m_MaxDistance = 2;
  m_pTempImage1 = NULL;
}

ShapeDetect::~ShapeDetect()
{
  if (m_pTempImage1 != NULL)
    cvReleaseImage(&m_pTempImage1);
}

void ShapeDetect::set_display_enable(bool displayEnable)
{
  m_DisplayEnable = displayEnable;
}

void ShapeDetect::Init()
{
  m_ImageSize   = cvGetSize(m_pInputImage);

  m_ROI.x = 0;
  m_ROI.y = 0;
  m_ROI.width  = m_ImageSize.width;
  m_ROI.height = m_ImageSize.height;

  m_pTempImage1 = cvCreateImage(m_ImageSize, IPL_DEPTH_8U, 1);
}

void ShapeDetect::DetectEdges(IplImage *pSrcImage)
{
  // Replace with something faster
  // We want the output to be 255 when the input pixel is 255 and at least
  // one neighboring pixel is 0

  cvCanny(pSrcImage, m_pTempImage1, 10, 100, 3);
}

void ShapeDetect::ComputePointList(IplImage *pSrcImage, CvSeq *pointSeq)
{
  int row;
  int col;
  int x;
  int y;
  int srcWidth;
  int srcHeight;
  unsigned char pixelValue;
  CvPoint currentPoint;

  srcWidth  = m_ROI.width;
  srcHeight = m_ROI.height;

  // Compute edges, put image into m_pTempImage1

  // DetectEdges(pSrcImage);

  for (row = 0; row < srcHeight; row++)
  {
    for (col = 0; col < srcWidth; col++)
    {
      x = col + m_ROI.x;
      y = row + m_ROI.y;

      // pixelValue = CV_IMAGE_ELEM(m_pTempImage1, unsigned char, y, x);
      pixelValue = CV_IMAGE_ELEM(pSrcImage, unsigned char, y, x);

      if (pixelValue == 255)
      {
        currentPoint.x = col;
        currentPoint.y = row;
        cvSeqPush(pointSeq, &currentPoint);
      }
    }
  }
}

void ShapeDetect::ConvexHull(IplImage *pSrcImage,
                             IplImage *pDestImage,
                             CvPoint **ppVertices,
                             int &numVertices)
{
  int i;
  int hullCount;
  IplImage* img;
  CvMemStorage *storage;
  CvSeq *pointSeq;
  CvSeq* hull;
  CvPoint *vertices;

  // Init storage and arrays

  m_pInputImage = pSrcImage;

  if (m_FirstFrame)
  {
    Init();
    m_FirstFrame = false;
  }

  storage = cvCreateMemStorage();

  pointSeq = cvCreateSeq(CV_SEQ_KIND_GENERIC | CV_32SC2,
                         sizeof(CvContour),
                         sizeof(CvPoint),
                         storage);

  // Create a list of points from the positions of the nonzero pixels
  // in the input image that are edge pixels.  We know that interior
  // points will be in the convex hull.

  ComputePointList(pSrcImage, pointSeq);

  // Compute the convex hull

  hull = cvConvexHull2(pointSeq, 0, CV_CLOCKWISE, 0);

  if (hull != NULL)
    hullCount = hull->total;
  else
    hullCount = 0;

  if (m_DisplayEnable)
  {
    // If the convex hull is NULL, the output image is identically zero
#if 1
    // Fill the area of the convex hull

    ConvexHullFill(hull, pDestImage);
#else
    // Draw the outline of the convex hull

    ConvexHullOutline(hull, pDestImage);
#endif
  }

  // Create a CvPoint array of the vertices

  vertices = new CvPoint[hullCount];

  for (i = 0; i < hullCount; i++)
    vertices[i] = **CV_GET_SEQ_ELEM(CvPoint*, hull, i);

  *ppVertices = vertices;
  numVertices = hullCount;

  // Cleanup

  cvClearMemStorage(storage);
  cvReleaseMemStorage(&storage);
}

void ShapeDetect::ConvexHullFill(CvSeq *convexHull,
                                 IplImage *pDestImage)
{
  int i;
  int hullCount;
  CvPoint *hullVertices;

  // Zero the display image

  cvZero(pDestImage);

  // If convex hull is NULL, return

  if (convexHull == NULL)
    return;

  // Fill the area of the convex hull

  hullCount = convexHull->total;
  hullVertices = new CvPoint[hullCount];

  for (i = 0; i < hullCount; i++)
    hullVertices[i] = **CV_GET_SEQ_ELEM(CvPoint*, convexHull, i);

  cvFillConvexPoly(pDestImage,
                   hullVertices,
                   hullCount,
                   cvScalar(255),
                   8,
                   0);

  delete[] hullVertices;
}

void ShapeDetect::ConvexHullOutline(CvSeq *convexHull,
                                    IplImage *pDestImage)
{
  int i;
  int hullCount;
  CvPoint currentPoint;
  CvPoint lastPoint;

  // Zero the display image

  cvZero(pDestImage);

  // If convex hull is NULL, return

  if (convexHull == NULL)
    return;

  // Draw the outline of the convex hull

  hullCount = convexHull->total;

  lastPoint = **CV_GET_SEQ_ELEM(CvPoint*, convexHull, hullCount - 1);

  for (i = 0; i < hullCount; i++)
  {
    currentPoint = **CV_GET_SEQ_ELEM(CvPoint*, convexHull, i);
    cvLine(pDestImage, lastPoint, currentPoint, cvScalar(255), 1, 8, 0);
    lastPoint = currentPoint;
  }
}

// Label the blobs in a binary image
// This is done in-place on the input image, 
// The ith connected region of 1s is filled with the ith label.
// The input image has values 0 for background and 0.5 for foreground.
// the output image has values 0 for background and 1 to N for blobs

void ShapeDetect::LabelBinaryBlobs(IplImage *pSrcImage, 
                                   CvConnectedComp *pBlobStats,
                                   int *numBlobsFound)
{
  CvPoint seedPoint;

  int i;
  float pixelValue;
  int Label;
  int numBlobs;
  int srcWidth;
  int srcHeight;
  int x;
  int y;
  int row;
  int col;
  int flags;

  m_pInputImage = pSrcImage;

  if (m_FirstFrame)
  {
    Init();
    m_FirstFrame = false;
  }

  srcWidth  = m_ROI.width;
  srcHeight = m_ROI.height;

  flags = 8;    // This is the number of pixel neighbors used to define
                // connectedness - either 4 or 8
  Label = 1;
  numBlobs = 0;

  // Main loop to label the blobs

  for (row = 0; row < srcHeight; row++)
  {
    for (col = 0; col < srcWidth; col++)
    {
      x = col + m_ROI.x;
      y = row + m_ROI.y;

      pixelValue = CV_IMAGE_ELEM(pSrcImage, float, y, x);

      if (fabs(pixelValue - 0.5) < 0.01)
      {
        seedPoint.x = col;
        seedPoint.y = row;

        cvFloodFill(pSrcImage,
                    seedPoint,
                    cvScalarAll((float) Label),
                    cvScalarAll(0),
                    cvScalarAll(0),
                    &(pBlobStats[numBlobs]),
                    flags,
                    NULL);

        numBlobs++;

        Label++;

        if (numBlobs == MAX_NUM_BLOBS)
        {
          *numBlobsFound = numBlobs;
          return;
        }
      }
    }
  }


  // Return the number of blobs

  *numBlobsFound = numBlobs;
}

// Label the blobs in a continuous-valued image
// This is done in-place on the input image, 
// The ith connected region of 1s is filled with the ith label.
// The input image has values 0 for background and > 0 for foreground.
// the output image has values 0 for background and 1 to N for blobs
// In the processing, the labels are negative, but the sign of the
// image is inverted at the end

void ShapeDetect::LabelContinuousBlobs(IplImage *pSrcImage, 
                                       CvConnectedComp *pBlobStats,
                                       int *numBlobsFound,
                                       double fgMinValue,
                                       double strokePixelDiff,
                                       int maxNumBlobs)
{
  CvPoint seedPoint;

  int i;
  float pixelValue;
  int Label;
  int numBlobs;
  int srcWidth;
  int srcHeight;
  int x;
  int y;
  int row;
  int col;
  int flags;

  m_pInputImage = pSrcImage;

  if (m_FirstFrame)
  {
    Init();
    m_FirstFrame = false;
  }

  srcWidth  = m_ROI.width;
  srcHeight = m_ROI.height;

#if 0
  strokePixelDiff  = 5.0;
  fgMinValue = 2.0;
#endif

  flags = 8;    // This is the number of pixel neighbors used to define
                // connectedness - either 4 or 8
  Label = 1;
  numBlobs = 0;

  // Main loop to label the blobs

  for (row = 0; row < srcHeight; row++)
  {
    for (col = 0; col < srcWidth; col++)
    {
      x = col + m_ROI.x;
      y = row + m_ROI.y;

      pixelValue = CV_IMAGE_ELEM(pSrcImage, float, y, x);

      if (pixelValue > fgMinValue)    // it is a foreground pixel
      {
        seedPoint.x = col;
        seedPoint.y = row;

        cvFloodFill(pSrcImage,
                    seedPoint,
                    cvScalarAll((float) -Label),  // fill value is < 0 here
                    cvScalarAll(strokePixelDiff),
                    cvScalarAll(strokePixelDiff),
                    &(pBlobStats[numBlobs]),
                    flags,
                    NULL);

        numBlobs++;

        Label++;

        if (numBlobs == maxNumBlobs)
          goto end_label_loop;
      }
    }
  }

end_label_loop:

  // Reverse the sign of the image

  cvConvertScale(pSrcImage, pSrcImage, -1.0, 0);

  // Clip all negative values to 0

  cvThreshold(pSrcImage, pSrcImage, 0, 0, CV_THRESH_TOZERO);

  // Return the number of blobs

  *numBlobsFound = numBlobs;
}

// Reduces the number of edges in a convex hull by 
// searching for nearly-collinear vertices

void ShapeDetect::ReduceConvexHull(CvPoint *pInputVertices,
                                   int numInputVertices,
                                   CvPoint **ppOutputVertices,
                                   int &numOutputVertices)
{
  Segment segment[numInputVertices];

  int j;
  int k;
  int startIndex;
  int nextIndex;
  double diffAngle;
  double diffAngleThreshold;
  int mergeCount;

  // Create an array of segment structures for the edges of the input polygon

  FillSegmentArray(pInputVertices,
                   numInputVertices,
                   segment);

  // Compare angles of consecutive edges
  // If the diff angle is small enough, we disable the second edge

  diffAngleThreshold = 1.0;
  mergeCount = 0;

  for (startIndex = 0; startIndex < numInputVertices; startIndex++)
  {
    nextIndex = (startIndex + 1) % numInputVertices;

    diffAngle = segment[nextIndex].angle - segment[startIndex].angle;

    if (fabs(diffAngle) < diffAngleThreshold)
    {
      segment[nextIndex].select = false;
      mergeCount++;
    }
  }

  // Write the remaining vertices to the output

  numOutputVertices = numInputVertices - mergeCount;
  *ppOutputVertices = new CvPoint[numOutputVertices];

  j = 0;

  for (startIndex = 0; startIndex < numInputVertices; startIndex++)
  {
    if (segment[startIndex].select)
    {
      (*ppOutputVertices)[j] = pInputVertices[startIndex];
      j++;
    }
  }
}

void ShapeDetect::FindLongDiagonal(CvPoint *convexHull, 
                                   int numVertices,
                                   int *pVertexIndices)
{
  int i;
  int j;
  double x[2];
  double y[2];
  double deltaX;
  double deltaY;
  double distance;
  double maxDistance;

  // We find the two vertices of the convex hull with the highest
  // distance between them.

  maxDistance = -1;

  for (i = 0; i < numVertices - 1; i++)
  {
    x[0] = convexHull[i].x;
    y[0] = convexHull[i].y;

    for (j = i; j < numVertices; j++)
    {
      x[1] = convexHull[j].x;
      y[1] = convexHull[j].y;

      deltaX = x[1] - x[0];
      deltaY = y[1] - y[0];

      distance = deltaX * deltaX + deltaY * deltaY;

      if (distance > maxDistance)
      {
        maxDistance = distance;
        pVertexIndices[0] = i;
        pVertexIndices[1] = j;
      }

    }
  }
}

void ShapeDetect::FindShortDiagonal(CvPoint *convexHull,
                                    int numVertices,
                                    int *pLongVertexIndices,
                                    int *pShortVertexIndices)
{
  int i;
  double x[2];
  double y[2];
  double distance;
  double maxPosDistance;
  double minNegDistance;
  double A;
  double B;
  double C;

  // Given the vertices for the longest diagonal
  // 1) we get a formula for the line between them
  // 2) we find the vertices on either side of the line
  //    with the maximal perpendicular distances to the line

  // Compute line between start and end vertices

  // Get the coordinates of the endpoints

  x[0] = convexHull[pLongVertexIndices[0]].x;
  y[0] = convexHull[pLongVertexIndices[0]].y;

  x[1] = convexHull[pLongVertexIndices[1]].x;
  y[1] = convexHull[pLongVertexIndices[1]].y;

  // Note that the line is given by
  //
  // (x - x0) * (y1 - y0) - (y - y0) * (x1 - x0) = 0
  //
  // and that this works for horizontal and vertical lines
  // We reduce the formula to A * x + B * y + C = 0

  A = y[1] - y[0];
  B = x[0] - x[1];
  C = x[1] * y[0] - x[0] * y[1];

  // Normalization not needed for comparison of distances
#if 0
  double scaleFactor;
  scaleFactor = 1.0 / sqrt(A * A + B * B);

  A = scaleFactor * A;
  B = scaleFactor * B;
  C = scaleFactor * C;
#endif

  // With A, B, and C so normalized. the perpendicular distance of a
  // point (x, y) to the line is abs(A * x + B * y + C); the sign
  // of A * x + B * y + C depends on which side of the line
  // the point is located.

  maxPosDistance = 0;
  minNegDistance = 0;

  for (i = 0; i < numVertices; i++)
  {
    x[0] = convexHull[i].x;
    y[0] = convexHull[i].y;

    distance = A * x[0] + B * y[0] + C;

    if (distance < minNegDistance)
    {
      minNegDistance = distance;
      pShortVertexIndices[0] = i;
    }

    if (distance > maxPosDistance)
    {
      maxPosDistance = distance;
      pShortVertexIndices[1] = i;
    }
  }

}

void ShapeDetect::FindCenter(CvPoint *convexHull,
                             int numVertices,
                             int *pLongVertexIndices,
                             int *pShortVertexIndices,
                             double *center)
{
  int i;
  double x[2];
  double y[2];
  double coeffLong[3];
  double coeffShort[3];

  // Get equation of long diagonal

  for (i = 0; i < 2; i++)
  {
    x[i] = convexHull[pLongVertexIndices[i]].x;
    y[i] = convexHull[pLongVertexIndices[i]].y;
  }

  GetLine(x, y, coeffLong);

  // Get equation of short diagonal

  for (i = 0; i < 2; i++)
  {
    x[i] = convexHull[pShortVertexIndices[i]].x;
    y[i] = convexHull[pShortVertexIndices[i]].y;
  }

  GetLine(x, y, coeffShort);

  // Get center

  GetIntersection(coeffLong, coeffShort, center[0], center[1]);
}

void ShapeDetect::FindBestEdge(Segment *segment,
                               int numInputVertices,
                               int startIndex,
                               int endIndex,
                               int &bestIndex)
{
  int index;
  double length;
  double maxLength;

  // Init

  maxLength = -1;
  bestIndex = -1;

  // Main loop

  index = startIndex;

  while (index != endIndex)
  {
    length = segment[index].length;

    if (length > maxLength)
    {
      maxLength = length;
      bestIndex = index;
    }

    index = (index + 1) % numInputVertices;
  }
}

bool ShapeDetect::GetEnclosingQuad(CvPoint *pInputVertices,
                                   int numInputVertices,
                                   CvPoint *pOutputVertices)
{
  int i;
  int j;
  int startIndex;
  int endIndex;
  int indexDelta;
  int bottomRightIndex;
  Segment segment[numInputVertices];
  Segment cornerEdges[4];
  Segment bestEdges[4];
  int longDiagEndpoints[2];
  int shortDiagEndpoints[2];
  int diagEndpoints[4];
  int bestIndex[4];
  double center[2];

  double x[4];
  double y[4];
  double coeff[4][3];

  // If there are less than 4 input vertices, return false

  if (numInputVertices < 4)
    return false;

  // Create an array of segment structures for the edges of the input polygon

  FillSegmentArray(pInputVertices,
                   numInputVertices,
                   segment);

  // Find the endpoints of the longest diagonal
  // Actually, we find the two vertices with the
  // longest distance between them.  For a
  // convex hull that is vaguely rectangular,
  // the two vertices should describe a diagonal

  FindLongDiagonal(pInputVertices,
                   numInputVertices,
                   longDiagEndpoints);

  // If the "diagonal" vertices are adjacent, they
  // are the endpoints of an edge, and there is no
  // short diagonal

  indexDelta = longDiagEndpoints[1] - longDiagEndpoints[0];
  indexDelta = indexDelta % numInputVertices;
  indexDelta = abs(indexDelta);

  if (indexDelta == 1)
    return false;

  // Find the endpoints of the shortest diagonal

  FindShortDiagonal(pInputVertices,
                    numInputVertices,
                    longDiagEndpoints,
                    shortDiagEndpoints);

  // Find the center - the intersection of the two
  // diagonals

  FindCenter(pInputVertices,
             numInputVertices,
             longDiagEndpoints,
             shortDiagEndpoints,
             &center[0]);

  // Merge the endpoints into one array

  diagEndpoints[0] = longDiagEndpoints[0];
  diagEndpoints[1] = longDiagEndpoints[1];
  diagEndpoints[2] = shortDiagEndpoints[0];
  diagEndpoints[3] = shortDiagEndpoints[1];

  // Find the corresponding edges (segments) that
  // start at these points

  for (i = 0; i < 4; i++)
    cornerEdges[i] = segment[diagEndpoints[i]];

  // Sort these according to start index

  qsort(cornerEdges,
        4,
        sizeof(Segment),
        CompareSegmentIndices);

  // Find best candidate edges between consecutive corners

  for (i = 0; i < 4; i++)
  {
    j = (i + 1) % 4;

    startIndex = cornerEdges[i].startIndex;
    endIndex   = cornerEdges[j].startIndex;

    FindBestEdge(segment, 
                 numInputVertices,
                 startIndex,
                 endIndex,
                 bestIndex[i]);

    bestEdges[i] = segment[bestIndex[i]];
  }

  // Sort these according to angle

  qsort(bestEdges,
        4,
        sizeof(Segment),
        CompareSegmentAngles);

  // Compute the 4 intersection points obtained by extending neighboring
  // edges to meet.

  // For each edge, get the edge endpoints
  // and compute the line through these

  for (i = 0; i < 4; i++)
  {
    startIndex = bestEdges[i].startIndex;
    endIndex   = bestEdges[i].endIndex;

    x[0] = pInputVertices[startIndex].x; 
    y[0] = pInputVertices[startIndex].y; 

    x[1] = pInputVertices[endIndex].x; 
    y[1] = pInputVertices[endIndex].y; 

    GetLine(x, y, &coeff[i][0]);
  }

  // Compute the intersections of consecutive lines

  for (i = 0; i < 4; i++)
  {
    j = (i + 1) % 4;

    GetIntersection(coeff[i], coeff[j], x[i], y[i]);
  }

  // Find the index of the botttom right corner

  bottomRightIndex = -1;

  for (i = 0; i < 4; i++)
  {
    if ((x[i] > center[0]) &&
        (y[i] > center[1]))
    {
      bottomRightIndex = i;
      break;
    }
  }

  // Copy the start vertices to the output

  for (i = 0; i < 4; i++)
  {
    j = (i + bottomRightIndex) % 4;
    pOutputVertices[i].x = (int) floor(x[j] + 0.5);
    pOutputVertices[i].y = (int) floor(y[j] + 0.5);
  }

  return true;
}

void ShapeDetect::FillSegmentArray(CvPoint *pInputVertices,
                                   int numInputVertices,
                                   Segment *segment)
{
  int startIndex;
  int endIndex;
  double Length;
  double Angle;
  double x[2];
  double y[2];
  double deltaX;
  double deltaY;
  double scaleFactor;
  scaleFactor = 180.0 / M_PI;

  for (startIndex = 0; startIndex < numInputVertices; startIndex++)
  {
    endIndex = (startIndex + 1) % numInputVertices;
    segment[startIndex].startIndex = startIndex;
    segment[startIndex].endIndex   = endIndex;
    segment[startIndex].select = true;
    segment[startIndex].rank = -1;

    // Compute length and angle

    x[0] = pInputVertices[startIndex].x;
    y[0] = pInputVertices[startIndex].y;

    x[1] = pInputVertices[endIndex].x;
    y[1] = pInputVertices[endIndex].y;

    deltaX = x[1] - x[0];
    deltaY = y[1] - y[0];

    Length = sqrt(deltaX * deltaX + deltaY * deltaY);
    Angle  = scaleFactor * atan2(deltaY, deltaX);

    segment[startIndex].length = Length;
    segment[startIndex].angle  = Angle;
  }
}

bool ShapeDetect::LineTest(CvPoint *vertices,
                           int startIndex,
                           int endIndex,
                           int numVertices)
{
  CvPoint currentPoint;
  double x, x0, x1;
  double y, y0, y1;
  double distance;

  // Line parameters such that Ax + By + C = 0

  double A;
  double B;
  double C;
  double scaleFactor;
  double Anorm;
  double Bnorm;
  double Cnorm;

  int index;

  // Compute line between start and end vertices

  // Get the coordinates of the endpoints

  x0 = vertices[startIndex].x;
  y0 = vertices[startIndex].y;

  x1 = vertices[endIndex].x;
  y1 = vertices[endIndex].y;

  // Note that the line is given by
  //
  // (x - x0) * (y1 - y0) - (y - y0) * (x1 - x0) = 0
  //
  // and that this works for horizontal and vertical lines

  A = y1 - y0;
  B = x0 - x1;
  C = x1 * y0 - x0 * y1;

  scaleFactor = 1.0 / sqrt(A * A + B * B);

  Anorm = scaleFactor * A;
  Bnorm = scaleFactor * B;
  Cnorm = scaleFactor * C;

  // Loop to test distances to line segment for intervening vertices

  index = (startIndex + 1) % numVertices;

  while (index != endIndex)
  {
    x = vertices[index].x;
    y = vertices[index].y;

    distance = fabs(Anorm * x + Bnorm * y + Cnorm);

    if (distance > m_MaxDistance)
      return false;

    index = (index + 1) % numVertices;
  }

  return true;
}

void ShapeDetect::GetLine(double *x,
                          double *y, 
                          double *coeff)
{
  coeff[0] = y[1] - y[0];
  coeff[1] = x[0] - x[1];
  coeff[2] = x[1] * y[0]- x[0] * y[1];
}

void ShapeDetect::GetIntersection(double *coeff0, 
                                  double *coeff1, 
                                  double &x, 
                                  double &y)
{
  int i;
  int j;

  double vector[2];
  double output[2];
  double matrix[2][2];
  double inverse[2][2];
  double detVal;

  // Read input values into matrix and vector

  vector[0] = coeff0[2];
  vector[1] = coeff1[2];

  for (j = 0; j < 2; j++)
  {
    matrix[0][j] = coeff0[j];
    matrix[1][j] = coeff1[j];
  }

  // Compute determinant

  detVal = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];

  // Compute inverse matrix

  inverse[0][0] = matrix[1][1];
  inverse[1][1] = matrix[0][0];
  inverse[0][1] = -matrix[0][1];
  inverse[1][0] = -matrix[1][0];

  detVal = 1.0 / detVal;

  for (i = 0; i < 2; i++)
  {
    for (j = 0; j < 2; j++)
      inverse[i][j] *= detVal;
  }

  // Compute output

  for (i = 0; i < 2; i++)
  {
    output[i] = 0;

    for (j = 0; j < 2; j++)
      output[i] += inverse[i][j] * vector[j];
  }

  x = -output[0];
  y = -output[1];
}

void ShapeDetect::SetROI(CvRect roi)
{
  m_ROI = roi;

  cvSetImageROI(m_pTempImage1, m_ROI);
}

void ShapeDetect::ResetROI()
{
  cvResetImageROI(m_pTempImage1);
}

int CompareSegmentAngles(const void *pSegment1, const void *pSegment2)
{
  Segment *pSegmentA;
  Segment *pSegmentB;
  double angleA;
  double angleB;

  pSegmentA = (Segment *) pSegment1;
  pSegmentB = (Segment *) pSegment2;

  angleA = pSegmentA->angle;
  angleB = pSegmentB->angle;

  if (angleA > angleB)
    return 1;
  else if (angleA < angleB)
    return -1;
  else
    return 0;
}

int CompareSegmentLengths(const void *pSegment1, const void *pSegment2)
{
  Segment *pSegmentA;
  Segment *pSegmentB;
  double lengthA;
  double lengthB;

  pSegmentA = (Segment *) pSegment1;
  pSegmentB = (Segment *) pSegment2;

  lengthA = pSegmentA->length;
  lengthB = pSegmentB->length;

  if (lengthA > lengthB)
    return 1;
  else if (lengthA < lengthB)
    return -1;
  else
    return 0;
}

int CompareSegmentIndices(const void *pSegment1, const void *pSegment2)
{
  Segment *pSegmentA;
  Segment *pSegmentB;
  double indexA;
  double indexB;

  pSegmentA = (Segment *) pSegment1;
  pSegmentB = (Segment *) pSegment2;

  indexA = pSegmentA->startIndex;
  indexB = pSegmentB->startIndex;

  if (indexA > indexB)
    return 1;
  else if (indexA < indexB)
    return -1;
  else
    return 0;
}
