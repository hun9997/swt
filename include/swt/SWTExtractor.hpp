// SWTExtractor.hpp
//
// Peter Wendt
// Gracenote LLC
// August 19, 2015
//

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "gnim.hpp"
#include "papyrus.hpp"
#include "Timer.h"
#include <tesseract/baseapi.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#define EXTRACTOR_LOG "SWTExtractor.log"

namespace papyrus
{
  class SWTExtractor : public TextExtractor
  {
   public:

    SWTExtractor();

    virtual ~SWTExtractor();

    void SetMaskRootName(std::string maskRootName);

    virtual void operator()(const gnim::Image& image,
                            papyrus::iterator<papyrus::bounding_box_t> text_boxes_begin,
                            papyrus::iterator<papyrus::bounding_box_t> text_boxes_end,
                            papyrus::inserter<papyrus::image_text_t> image_text);

   private:

    tesseract::TessBaseAPI m_TextRecognizer;
    Timer m_Timer;

    std::string   m_MaskRootName;
    std::ofstream m_LogFile;

    IplImage *m_pBoxImageIpl;
    IplImage *m_pTempImage;

    CvFont   m_TextFont;
    CvScalar m_TextColor;

    double m_GuardBandRatio;
    int    m_ResizedHeight;

    std::vector<cv::Mat>         m_WordMasks;
    std::vector<cv::RotatedRect> m_MinBlobBoxes;

    IplImage *m_pLabelledImageIpl;
    cv::Mat   m_LabelledImageMat;

    void SaveMaskImages();
    void GetMask(int i, cv::Mat &textMask);
    void GetText(int i, std::string &extractedText, bool &validText);
    bool ValidText(const char *charString);
    void DrawBoxImage();
    void InitText();
    void DrawText(std::string textString,
                  cv::RotatedRect minBox);
    void DisplayBoxImage();
  };
}
