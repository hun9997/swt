#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "gnim.hpp"
#include "papyrus.hpp"

void IplToGNIM(IplImage *pSrcImage,
               gnim::Image **pDestImage);

void GNIMToIpl(const gnim::Image *pSrcImage,
               IplImage **pDestImage);

void InvertGNIM(const gnim::Image *pSrcImage,
                gnim::Image **pDestImage);

void RectToBox(cv::Rect &rect, papyrus::bounding_box_t &box);

void BoxToRect(papyrus::bounding_box_t &box, cv::Rect &rect);

void RotatedRectToBox(cv::RotatedRect &rect, papyrus::bounding_box_t &box);

void BoxToRotatedRect(papyrus::bounding_box_t &box, cv::RotatedRect &rect);
