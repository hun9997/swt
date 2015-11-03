#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>

void subtractBG(IplImage *pSrcImage,
                IplImage *pDestImage,
                int windowSize);

void getTextMask(IplImage *pSWTImage,
                 IplImage *pMaskImage);

void FindBoxes(IplImage *pLabelledBlobImage,
               std::vector<cv::Rect> &blobBoxes);

void FindBoxesAndMasks(IplImage *pLabelledBlobImage,
                       std::vector<cv::Rect> &blobBoxes,
                       std::vector<cv::Mat> &blobMasks);

void DrawBoxes(IplImage *pDisplayImage,
               std::vector<cv::Rect> blobBoxes);

void DrawBox(IplImage *pDisplayImage,
             cv::Rect &blobBox);

void FindMinBoxes(std::vector<cv::Mat> &blobMasks,
                  std::vector<cv::Rect> &blobBoxes,
                  std::vector<cv::RotatedRect> &blobMinBoxes);

void DrawMinBoxes(IplImage *pDisplayImage,
                  std::vector<cv::RotatedRect> &blobBoxes,
                  int special = 0);

void DrawMinBox(IplImage *pDisplayImage,
                cv::RotatedRect &blobBox,
                int special = 0);

void GetFlatRegions(IplImage *pOrigImage,
                    IplImage *pLabelledBlobs,
                    CvConnectedComp *pBlobStats,
                    int numBlobs,
                    float flatThreshold,
                    IplImage *pOutputMask);

void SuppressSmallRegions(CvConnectedComp *pBlobStats,
                          int numBlobs,
                          int minSize,
                          bool *pBlobSelect);

void PadRegion(CvRect &inputROI,
               CvSize &imageSize,
               int padWidth,
               CvRect &outputROI);

void GetBlobMask(IplImage *pBlobImage,
                 CvConnectedComp *pBlobStats,
                 int blobIndex,
                 cv::Mat &blobMask,
                 int dilateRadius = 0,
                 int borderWidth = 0);

void GetBlobMask2(IplImage *pBlobImage,
                  int blobIndex,
                  cv::Rect &blobBox,
                  cv::Mat &blobMask,
                  int dilateRadius = 0,
                  int borderWidth = 0);

void GetBlobMasks(IplImage *pBlobImage,
                  std::vector<cv::Rect> &blobBoxes,
                  std::vector<cv::Mat> &blobMasks);

float GetMaxStrokeWidth(IplImage *pSWTImage,
                        IplImage *pBlobImage,
                        CvConnectedComp *pBlobStats,
                        int blobIndex);

void SuppressThickStrokes(IplImage *pSWTImage,
                          IplImage *pBlobImage,
                          CvConnectedComp *pBlobStats,
                          int numBlobs,
                          int maxStrokeWidth,
                          bool *pBlobSelect);

void SuppressLargeRegions(CvConnectedComp *pBlobStats,
                          int numBlobs,
                          int maxStrokeWidth,
                          bool *pBlobSelect);

void ClearBlob(IplImage *pBlobImage,
               CvConnectedComp *pBlobStats,
               int blobIndex);

void ClearBlobs(IplImage *pBlobImage,
                CvConnectedComp *pBlobStats,
                int numBlobs,
                bool *pBlobSelect);

void Normalize(IplImage *pSrcImage,
               IplImage *pDestImage);

void Outline(IplImage *pSrcImage,
             IplImage *pDestImage);
