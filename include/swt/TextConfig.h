#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <vector>

struct TextConfig
{
  // Modes

  bool darkText;
  bool lightText;
  bool logSWT;
  bool smoothSWT;
  bool separateBlobs;
  bool mergeBlobs;
  bool suppressEdges;
  bool padRegions;
  bool suppressSmallRegions;
  bool suppressThickStrokes;
  bool displayImages;
  bool saveImage;

  // Parameters

  CvSize imageSize;

  int maxNumBlobs;
  float angleRangeDivisor;
  float minStrokeWidthDivisor;
  float maxStrokeWidthDivisor;
  int smoothFilterSize; // 3 or 5
  int separateFilterSize;
  int mergeFilterSize;
  int dilateRadius;
  int padWidth;
  double cannyLowThreshold;
  double cannyHighThreshold;
  double fgMinValue;
  double strokePixelDiff;
};

void SetTextConfig(TextConfig &config);
