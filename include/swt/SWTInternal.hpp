#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

/*
 * Copyright 2012 Andrew Perrault and Saurav Kumar.
 * 
 * This file is part of DetectText.
 * 
 * DetectText is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 * 
 * DetectText is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * DetectText.  If not, see <http://www.gnu.org/licenses/>.
 */

struct SWTPoint2d
{
  int x;
  int y;
  float SWT;
};

struct Ray
{
  SWTPoint2d p;
  SWTPoint2d q;
  std::vector<SWTPoint2d> points;
};

class SWTInternal
{
 public:

  SWTInternal();

  ~SWTInternal();

  void getTransform(IplImage *input,
                    IplImage *SWTImage,
                    bool dark_on_light,
                    double threshold_low,
                    double threshold_high,
                    int angle_divisor);

 private:

  IplImage *grayImage;
  IplImage *edgeImage;
  IplImage *gaussianImage;
  IplImage *gradientX;
  IplImage *gradientY;

  void preprocess(double threshold_low, double threshold_high);

  void basicTransform(IplImage *edgeImage,
                      IplImage *gradientX,
                      IplImage *gradientY,
                      bool dark_on_light,
                      int angle_divisor,
                      IplImage *SWTImage,
                      std::vector<Ray> &rays);

  void SWTMedianFilter(IplImage *SWTImage, std::vector<Ray> &rays);
  
  void CreateImages(CvSize imageSize);

  void DeleteImages();
};

bool Point2dSort(const SWTPoint2d &lhs, const SWTPoint2d &rhs);
