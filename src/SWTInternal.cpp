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

#include <cassert>
#include <cmath>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <math.h>
#include <time.h>
#include <utility>
#include <algorithm>
#include <vector>
#include <SWTInternal.hpp>

#define PI 3.14159265

using namespace std;
using namespace cv;

SWTInternal::SWTInternal()
{
}

SWTInternal::~SWTInternal()
{
}

void SWTInternal::getTransform(IplImage *input,
                               IplImage *SWTImage,
                               bool dark_on_light,
                               double threshold_low,
                               double threshold_high,
                               int angle_divisor)
{
  vector<Ray> rays;

  CvSize inputSize;

  // Create IplImages

  inputSize = cvGetSize(input); 

  CreateImages(inputSize);

  // Convert to grayscale

  if (input->nChannels == 3)
    cvCvtColor(input, grayImage, CV_RGB2GRAY);
  else if (input->nChannels == 1)
    cvCopy(input, grayImage);

  preprocess(threshold_low, threshold_high);

  // Calculate SWT and return ray vectors

  cvSet(SWTImage, cvScalarAll(-1));

  basicTransform(edgeImage,
                 gradientX,
                 gradientY,
                 dark_on_light,
                 angle_divisor,
                 SWTImage,
                 rays);

  SWTMedianFilter(SWTImage, rays);

  // Cleanup

  cvReleaseImage(&edgeImage);
}

void SWTInternal::preprocess(double threshold_low, double threshold_high)
{
  // Create Canny Image

  cvCanny(grayImage, edgeImage, threshold_low, threshold_high, 3);

  // Create gradient X, gradient Y

  cvConvertScale(grayImage, gaussianImage, 1.0 / 255.0, 0);
  cvSmooth(gaussianImage, gaussianImage, CV_GAUSSIAN, 5, 5);

  cvSobel(gaussianImage, gradientX, 1, 0, CV_SCHARR);
  cvSobel(gaussianImage, gradientY, 0, 1, CV_SCHARR);

  cvSmooth(gradientX, gradientX, 3, 3);
  cvSmooth(gradientY, gradientY, 3, 3);
}

void SWTInternal::basicTransform(IplImage *edgeImage,
                                 IplImage *gradientX,
                                 IplImage *gradientY,
                                 bool dark_on_light,
                                 int angle_divisor,
                                 IplImage *SWTImage,
                                 vector<Ray> &rays)
{
  float prec;
  // Ray r;
  SWTPoint2d p;
  SWTPoint2d pnew;
  // vector<SWTPoint2d> points;
  vector<SWTPoint2d>::iterator pit;
  int row;
  int col;
  float curX;
  float curY;
  int   curPixX;
  int   curPixY;
  const uchar *ptr;
  float G_x;
  float G_y;
  float G_xt;
  float G_yt;
  float mag;
  float length;
  float deltaX;
  float deltaY;

  // First pass

  prec = .05;

  for (row = 0; row < edgeImage->height; row++)
  {
    ptr = (const uchar *)(edgeImage->imageData + row * edgeImage->widthStep);

    for (col = 0; col < edgeImage->width; col++)
    {
      // If not an edge pixel, we skip processing

      if (*ptr <= 0)
      {
        ptr++;
        continue;
      }

      Ray r;
      vector<SWTPoint2d> points;

      p.x = col;
      p.y = row;
      r.p = p;

      // points.clear();
      points.push_back(p);

      curX = (float) col + 0.5;
      curY = (float) row + 0.5;
      curPixX = col;
      curPixY = row;

      // Get the gradients

      G_x = CV_IMAGE_ELEM(gradientX, float, row, col);
      G_y = CV_IMAGE_ELEM(gradientY, float, row, col);

      // Normalize the gradients

      mag = sqrt((G_x * G_x) + (G_y * G_y));

      G_x = G_x / mag;
      G_y = G_y / mag;

      // For dark text on light background, invert the gradients

      if (dark_on_light)
      {
        G_x = -G_x;
        G_y = -G_y;
      }

      // Creep in the direction of the gradient until we find another
      // edge pixel.  Do the basic SWT logic.

      while (true)
      {
        // Creep a bit

        curX += G_x * prec;
        curY += G_y * prec;

        // If we have arrived at a new pixel

        if ((int) (floor(curX)) != curPixX || (int) (floor(curY)) != curPixY)
        {
          curPixX = (int) floor(curX);
          curPixY = (int) floor(curY);

          // Check if pixel is outside image boundary

          if (curPixX < 0 ||
             (curPixX >= SWTImage->width) ||
              curPixY < 0 ||
             (curPixY >= SWTImage->height))
          {
            break;
          }

          pnew.x = curPixX;
          pnew.y = curPixY;

          points.push_back(pnew);

          // If we are at an edge pixel, we are at the other end of the ray

          if (CV_IMAGE_ELEM(edgeImage, uchar, curPixY, curPixX) > 0)
          {
            r.q = pnew;

            // Get the gradients

            G_xt = CV_IMAGE_ELEM(gradientX, float, curPixY, curPixX);
            G_yt = CV_IMAGE_ELEM(gradientY, float, curPixY, curPixX);

            mag = sqrt((G_xt * G_xt) + (G_yt * G_yt));

            // Normalize the gradients

            G_xt = G_xt / mag;
            G_yt = G_yt / mag;

            // For dark text on light background, invert the gradients

            if (dark_on_light)
            {
              G_xt = -G_xt;
              G_yt = -G_yt;
            }

            // If the angle between the gradient vectors is small enough,
            // we fill the ray.
            // The min() ensures that, in the final image, we get the local
            // width, not the local length, at each stroke pixel.

            if (acos(G_x * -G_xt + G_y * -G_yt) < PI / (double) angle_divisor)
            {
              // Ray is valid; write it to the SWT image

              deltaX = (float) r.q.x - (float) r.p.x;
              deltaY = (float) r.q.y - (float) r.p.y;

              length = sqrt(deltaX * deltaX + deltaY * deltaY);

              for (pit = points.begin(); pit != points.end(); pit++)
              {
                // If the SWT pixel was empty, we fill it with the
                // length of the ray.  Otherwise, we us the minimum
                // of the new length and the previous value of the pixel.

                if (CV_IMAGE_ELEM(SWTImage, float, pit->y, pit->x)<0)
                {
                  CV_IMAGE_ELEM(SWTImage, float, pit->y, pit->x)= length;
                }
                else
                {
                  CV_IMAGE_ELEM(SWTImage, float, pit->y, pit->x) =
                    min(length, CV_IMAGE_ELEM(SWTImage, float, pit->y, pit->x));
                }
              }

              r.points = points;
              rays.push_back(r);

            } // end of valid-ray if() block

            break;
          } // end of if() for 2nd edge pixel
        }  // end of if() for incrementing to a new pixel location
      } // end of while(true) - the creeping while()

      ptr++;
    } // end of for(col) pixel loop
  } // end of for(row) row loop


} // end of the whole function

void SWTInternal::SWTMedianFilter(IplImage *SWTImage, vector<Ray> &rays)
{
  vector<Ray>::iterator rit;
  vector<SWTPoint2d>::iterator pit;
  float median;

  for (rit = rays.begin(); rit != rays.end(); rit++)
  {
    for (pit = rit->points.begin(); pit != rit->points.end(); pit++)
    {
      pit->SWT = CV_IMAGE_ELEM(SWTImage, float, pit->y, pit->x);
    }

    sort(rit->points.begin(), rit->points.end(), &Point2dSort);

    median = (rit->points[rit->points.size() / 2]).SWT;

    for (pit = rit->points.begin(); pit != rit->points.end(); pit++)
    {
      CV_IMAGE_ELEM(SWTImage, float, pit->y, pit->x)= min(pit->SWT, median);
    }
  }
}

void SWTInternal::CreateImages(CvSize imageSize)
{
  grayImage     = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
  edgeImage     = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
  gaussianImage = cvCreateImage(imageSize, IPL_DEPTH_32F, 1);
  gradientX     = cvCreateImage(imageSize, IPL_DEPTH_32F, 1);
  gradientY     = cvCreateImage(imageSize, IPL_DEPTH_32F, 1);
}

void SWTInternal::DeleteImages()
{
  cvReleaseImage(&gaussianImage);
  cvReleaseImage(&grayImage);
  cvReleaseImage(&gradientX);
  cvReleaseImage(&gradientY);
  cvReleaseImage(&edgeImage);
}

bool Point2dSort(const SWTPoint2d &lhs, const SWTPoint2d &rhs)
{
  return lhs.SWT < rhs.SWT;
}
