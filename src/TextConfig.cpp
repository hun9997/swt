#include "TextConfig.h"

void SetTextConfig(TextConfig &config)
{
  // Modes

  config.darkText      = true;
  config.lightText     = true;
  config.logSWT        = false;
  config.smoothSWT     = false;
  config.separateBlobs = false;
  config.mergeBlobs    = false;
  config.suppressEdges = true;
  config.padRegions    = false;
  config.suppressSmallRegions = false;
  config.suppressThickStrokes = false;
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
