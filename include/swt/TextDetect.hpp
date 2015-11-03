#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

#include "ShapeDetect.hpp"
#include "SWT.hpp"
#include "TextConfig.h"

class TextDetect
{
 public:

  TextDetect(TextConfig *textConfig);

  ~TextDetect();

  void GetText(IplImage *pSrcImage,
               std::vector<cv::Mat>  &textMasks,
               std::vector<cv::Rect> &textBoxes);

  void GetImage(int lightDark, char *type, IplImage *pDestImage);

  void GetSWTs(cv::Mat *srcImage,
               cv::Mat *swtImage[2]);

 private:

  bool isInitialized;

  // Images

  IplImage  *pInputImage;
  IplImage  *pEdgeImage;
  IplImage  *pTempImage;
  IplImage  *pSWTImage[2];
  IplImage  *pSWTMaskImage[2];
  IplImage  *pBlobImage[2];
  IplImage  *pTextBoxImage;
  IplImage  *pSWTDisplayImage[2];
  IplImage  *pBlobDisplayImage[2];

  CvSize     imageSize;

  // config structure

  TextConfig config;

  // Blobs

  ShapeDetect shapeDetect;
  CvConnectedComp *pBlobStats;
  bool *pBlobSelect;
  int numBlobsFound;

  // SWT

  SWT m_SWT;

  // Functions

  void SetDefaultConfig();

  void Init(int imageHeight);

  void InitImages();

  void DeleteImages();

  void InitWindows();

  void DisplayImages();

  void CreateBoxImage(std::vector<cv::Rect> &textBoxes);

  void PerfStats();

};
