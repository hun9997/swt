//
// ShapeDetect.hpp
//

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <stdio.h>

typedef struct
{
  int    startIndex;
  int    endIndex;
  double length;
  double angle;
  int    rank;
  bool   select;
} Segment;

class ShapeDetect
{
 public:

  ShapeDetect();

  ~ShapeDetect();

  void set_display_enable(bool displayEnable);

  void ConvexHull(IplImage *pSrcImage,
                  IplImage *pDestImage,
                  CvPoint **ppVertices,
                  int &numVertices);

  void LabelBinaryBlobs(IplImage *pSrcImage,
                        CvConnectedComp *pBlobStats,
                        int *numBlobsFound);

  void LabelContinuousBlobs(IplImage *pSrcImage,
                            CvConnectedComp *pBlobStats,
                            int *numBlobsFound,
                            double fgMinValue,
                            double strokePixelDiff,
                            int maxNumBlobs);

  void ReduceConvexHull(CvPoint *pInputVertices,
                        int numInputVertices,
                        CvPoint **ppOutputVertices,
                        int &numOutputVertices);

  bool GetEnclosingQuad(CvPoint *pInputVertices,
                        int numInputVertices,
                        CvPoint *pOutputVertices);

  void SetROI(CvRect roi);

  void ResetROI();
                        

 private:

  static const int MAX_NUM_BLOBS = 10000;

  bool m_DisplayEnable;
  bool   m_FirstFrame;
  CvSize m_ImageSize;
  int    m_MaxDistance;

  IplImage *m_pInputImage;
  IplImage *m_pTempImage1;

  CvRect m_ROI;

  void Init();

  void DetectEdges(IplImage *pSrcImage);

  void ComputePointList(IplImage *pSrcImage, CvSeq *pointSeq);

  void ConvexHullFill(CvSeq *convexHull, IplImage *pDestImage);

  void ConvexHullOutline(CvSeq *convexHull, IplImage *pDestImage);

  void FindLongDiagonal(CvPoint *convexHull, 
                        int numVertices,
                        int *pVertexIndices);

  void FindShortDiagonal(CvPoint *convexHull,
                         int numVertices,
                         int *pLongVertexIndices,
                         int *pShortVertexIndices);

  void FindCenter(CvPoint *convexHull,
                  int numVertices,
                  int *pLongVertexIndices,
                  int *pShortVertexIndices,
                  double *center);

  void FindBestEdge(Segment *segment,
                    int numVertices,
                    int startIndex,
                    int endIndex,
                    int &bestIndex);

  void FillSegmentArray(CvPoint *convexHull,
                        int numVertices,
                        Segment *segment);

  bool LineTest(CvPoint *vertices,
                int startIndex,
                int endIndex,
                int numVertices);

  void GetLine(double *x,
               double *y,
               double *coeff);

  void GetIntersection(double *coeff0,
                       double *coeff1,
                       double &x,
                       double &y);
};

int CompareSegmentAngles(const void *pSegment1, const void *pSegment2);

int CompareSegmentLengths(const void *pSegment1, const void *pSegment2);

int CompareSegmentIndices(const void *pSegment1, const void *pSegment2);
