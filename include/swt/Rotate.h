//
// Rotate.h
//  Rotation utilities
//
// Peter Wendt
// Apr. 17, 2014
// Zeitera LLC
//

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Size RotatedImageSize(cv::Size inputSize,
                          double angleDegrees);

cv::Mat* RotateImage(cv::Mat &inputImage,
                     double angleDegrees,
                     bool isBinary = false);

void AlignRegionMat(cv::Mat &inputImage,
                    cv::Mat &outputImage,
                    cv::RotatedRect &minBox,
                    bool isBinary = false);

void AlignRegionIpl(IplImage *inputImage,
                    IplImage *outputImage,
                    cv::RotatedRect &minBox);

void AlignPoints(std::vector<cv::Point2f> &inCorners,
                 std::vector<cv::Point2f> &outCorners,
                 cv::RotatedRect &minBox);

void AlignPoints2(std::vector<cv::Point2f> &inPoints,
                  std::vector<cv::Point2f> &outPoints,
                  cv::Point2f center,
                  double angle);
