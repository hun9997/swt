// FindWords.hpp
//
// class to extract and recognize text
//
// Peter Wendt
// Zeitera LLC
// May 7, 2015
//

#pragma once

#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "TextConfig.h"
#include "TextDetect.hpp"
#include "ImageUtils.h"
#include "Process.h"
#include "Rotate.h"

  struct CharLoc {
    int charIndex;
    cv::Point2f center;
  };

class FindWords
{
 public:

  FindWords();

  ~FindWords();

  void process(IplImage *pSrcImage,
               int lightDarkText,
               IplImage *pLabelledWordImage,
               std::vector <cv::Rect> &outWordBoxes);

 private:

  TextConfig   wordConfig;
  TextDetect  *pWordDetect;
  std::vector<cv::Rect> wordBoxes;
  std::vector<cv::RotatedRect> wordMinBoxes;
  std::vector<cv::RotatedRect> wordLowBoxes;
  std::vector<cv::Mat> wordMasks;
  std::vector<double> wordAngles;

  TextConfig   charConfig;
  TextDetect  *pCharDetect;
  std::vector<cv::Rect> charBoxes;
  std::vector<cv::RotatedRect> charMinBoxes;
  std::vector<cv::Mat>  charMasks;

  std::vector< std::vector <int> > charsInWords;
  std::vector< std::vector <cv::Mat> > charImages;

  // Images

  IplImage *pCharMaskImage;
  IplImage *pWordMaskImage;

  // Functions

  void Init(int lightDarkText);

  void SetWordConfig();

  void SetCharConfig();

  void CharBlobsInWordBlobs();

  bool IsCharInWord(cv::Mat &charMask,
                    cv::Rect &charRect,
                    cv::Mat &wordMask,
                    cv::Rect &wordRect);

  void LabelWords(IplImage *pLabelledWordImage);

  void GetWordAngles();

  void getWordAngle(int wordIndex, double &estWordAngle);

  double ConvertAngle(cv::RotatedRect &minBox);

  void SortCharsInWord(int wordIndex);

  void AlignLocations(std::vector<CharLoc> &inputLocs, 
                      std::vector<CharLoc> &outputLocs,
                      cv::Point2f wordCenter,
                      double wordAngle);

  void AlignCharImages();

  void SaveImages(IplImage *pSrcImage);

  IplImage* UprightBoxImage(IplImage *pSrcImage,
                            std::vector<cv::Rect> &textBoxes);

  IplImage* MinBoxImage(IplImage *pSrcImage,
                        std::vector<cv::RotatedRect> &textBoxes,
                        std::vector<cv::RotatedRect> &lowBoxes);

  void DisplayImages();
};
