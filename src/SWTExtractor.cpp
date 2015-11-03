#include "SWTExtractor.hpp"
#include "Rotate.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <tesseract/baseapi.h>
#include <ConvertUtils.h>
#include <Process.h>
#include <Timer.h>
#include <vector>
#include <string>
#include <sstream>
#include <cctype>

#define __SWT_EXTRACTOR_SAVE
#define __SWT_EXTRACTOR_DISPLAY
#define MIN_CONFIDENCE 50

using namespace std;
using namespace cv;
using namespace gnim;
using namespace papyrus;
using namespace tesseract;

SWTExtractor::SWTExtractor()
{
  double procTime;

  m_LogFile.open(EXTRACTOR_LOG, std::ofstream::out | std::ofstream::app);
  m_Timer.Start();

  // Init Tesseract OCR

  m_TextRecognizer.Init(NULL, "eng", tesseract::OEM_DEFAULT);
  m_TextRecognizer.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);

  procTime = m_Timer.Stop();
  m_LogFile << "Tesseract init time = " << procTime << endl;

  m_GuardBandRatio = 0.05;
  m_ResizedHeight  = 100;
}

SWTExtractor::~SWTExtractor()
{
  m_LogFile.close();
  m_TextRecognizer.End();
}

void SWTExtractor::SetMaskRootName(string maskRootName)
{
  m_MaskRootName = maskRootName;
}

void SWTExtractor::operator()(const gnim::Image& image,
                              papyrus::iterator<bounding_box_t> text_boxes_begin,
                              papyrus::iterator<bounding_box_t> text_boxes_end,
                              papyrus::inserter<image_text_t> image_text)
{
  int i;
  papyrus::iterator<bounding_box_t> itr;

  bounding_box_t tempBox;
  RotatedRect    tempRect;

  string extractedString;
  Mat textMask;
  image_text_t textPair;

  bool validText;

  double procTime;

  // Init

  m_MinBlobBoxes.clear();
  m_WordMasks.clear();

  m_Timer.Start();

  // Convert image from gnim format to IplImage and thence to Mat

  GNIMToIpl(&image, &m_pLabelledImageIpl);
  m_LabelledImageMat = Mat(m_pLabelledImageIpl, true);

  // Convert input boxes to RotatedRects

  for (itr = text_boxes_begin;
       itr != text_boxes_end;
       itr++)
  {
    tempBox = *itr;
    BoxToRotatedRect(tempBox, tempRect);

    m_MinBlobBoxes.push_back(tempRect);
  }

#ifdef __SWT_EXTRACTOR_DISPLAY
  // Create box image

  DrawBoxImage();
#endif

  // Find the masks amd align them

  for (i = 0; i < m_MinBlobBoxes.size(); i++)
  {
    GetMask(i, textMask);

    // Push the aligned mask to the output vector

    m_WordMasks.push_back(textMask);
  }

  procTime = m_Timer.Stop();

  m_LogFile << "time to find masks = " << procTime << endl;
  m_LogFile << "number of masks = " << m_WordMasks.size() << endl;

#ifdef __SWT_EXTRACTOR_SAVE
  m_Timer.Start();

  // Save the word masks as images

  SaveMaskImages();

  procTime = m_Timer.Stop();

  m_LogFile << "time to save masks = " << procTime << endl;
#endif

  // Set up for text labels

  InitText();

  // Send text images to OCR
  // Insert box/string pairs to output container
  // Remember that the papyrus inserter inserts in reverse order

  m_Timer.Start();

  i = 0;

  for (itr = text_boxes_begin;
       itr != text_boxes_end;
       itr++)
  {
    tempBox = *itr;

    // Get text from this mask

    GetText(i, extractedString, validText);

    // If text is not valid, go to next mask

    if (!validText)
    {
      i++;
      continue;
    }

#ifdef __SWT_EXTRACTOR_DISPLAY
    // Write recognized text next to blob

    DrawText(extractedString, m_MinBlobBoxes[i]);
#endif

    // Create the box/string pair

    textPair.first  = tempBox;
    textPair.second = extractedString;

    // Insert this pair in the output container

    image_text.insert(textPair);

    i++;
  }

  procTime = m_Timer.Stop();

  m_LogFile << "text recognition time = " << procTime << endl;

#ifdef __SWT_EXTRACTOR_DISPLAY
  DisplayBoxImage();
#endif

  // Cleanup

  cvReleaseImage(&m_pLabelledImageIpl);

  // Return

  return;
}

void SWTExtractor::SaveMaskImages()
{
  int i;
  std::stringstream stringStream;
  string saveFileName;
  IplImage saveImage;

  for (i = 0; i < m_WordMasks.size(); i++)
  {
    stringStream.str("");

    stringStream << m_MaskRootName << "_" << i << ".jpg";

    saveFileName = stringStream.str();

    saveImage = IplImage(m_WordMasks[i]);

    cvSaveImage(saveFileName.c_str(), &saveImage);
  }
}

void SWTExtractor::GetMask(int i, Mat &textMask)
{
  RotatedRect tempRect;
  int tempWidth;
  int tempHeight;
  int rotatedHeight;
  int rotatedWidth;
  double resizeRatio;
  int resizedWidth;
  int resizedHeight;
  Size resizedSize;

  // Setup

  // Get and pad the min bounding box

  tempRect = m_MinBlobBoxes[i];

  tempRect.size.width  *=  (1.0 + 2.0 * m_GuardBandRatio);
  tempRect.size.height *=  (1.0 + 2.0 * m_GuardBandRatio);

  tempHeight = (int) floor(tempRect.size.height + 0.5);
  tempWidth  = (int) floor(tempRect.size.width  + 0.5);

  // Set up the images

  Mat tempMask(tempHeight, tempWidth, CV_8UC1);
  Mat *pRotMask;

  // Align the rotated text

  AlignRegionMat(m_LabelledImageMat,
                 tempMask,
                 tempRect,
                 0);

  // If the output image is taller than it is wide, rotate it by
  // 90 degrees.  This will work for long words, but not for single
  // characters

  double angle;

#if 1
  if (tempWidth > tempHeight)
    angle = 0;
  else
    angle = 90;
#else
  angle = 0;
#endif

  // Rotation is counterclockwise
    
  pRotMask = RotateImage(tempMask, angle, 0);

  // Resized size for input to OCR

  rotatedWidth  = pRotMask->cols;
  rotatedHeight = pRotMask->rows;
   
  resizeRatio   = (double) m_ResizedHeight / (double) rotatedHeight;
  resizedHeight = m_ResizedHeight;
  resizedWidth  = (int) floor(resizeRatio * rotatedWidth + 0.5);
  resizedSize   = Size(resizedWidth, resizedHeight);

  // Set up the images

  Mat resizedMask(resizedHeight, resizedWidth, CV_8UC1);
  Mat diffMask(resizedHeight, resizedWidth, CV_8UC1);
  Mat threshMask(resizedHeight,  resizedWidth, CV_8UC1);

  // Resize it

  resize(*pRotMask, resizedMask, resizedSize, 0, 0, INTER_LINEAR);

  // Make it binary

  absdiff(resizedMask, Scalar(i + 1), diffMask);

  compare(diffMask, Scalar(0.5), threshMask, CMP_LT);

  delete pRotMask;

  textMask = threshMask;
}

void SWTExtractor::GetText(int i, string &extractedText, bool &validText)
{
  const char *extractedWord;
  float confidence;
  int wordCount;
  tesseract::ResultIterator *resultIter;
  tesseract::PageIteratorLevel pageLevel;

  // Send the binary text mask to OCR

  // Setup

  m_TextRecognizer.SetImage((unsigned char *) m_WordMasks[i].data,
                            m_WordMasks[i].cols,
                            m_WordMasks[i].rows,
                            1,
                            m_WordMasks[i].cols);

  m_TextRecognizer.Recognize(0);
  resultIter = m_TextRecognizer.GetIterator();
  pageLevel = tesseract::RIL_WORD;

  if (resultIter == 0)
  {
    validText = false;
    return;
  }

  extractedText = string("");

  // Get the text

  wordCount = 0;

  do
  {
    extractedWord = resultIter->GetUTF8Text(pageLevel);
    confidence = resultIter->Confidence(pageLevel);

    printf("confidence = %f\n", confidence);

    if ((confidence < MIN_CONFIDENCE) ||
        (extractedWord == NULL) ||
        (strlen(extractedWord) == 0))
    {
      validText = false;
      return;
    }

    if (wordCount == 0)
      extractedText = string(extractedWord);
    else
      extractedText = extractedText + " " + string(extractedWord);

    wordCount++;

    delete[] extractedWord;
  }
  while (resultIter->Next(pageLevel));
    
  // Decision to thin bad text
    
  validText = true;

#if 1
  validText = ValidText(extractedText.c_str());
#endif
}

void SWTExtractor::DrawBoxImage()
{
  CvSize inputSize;

  // Setup

  inputSize.width  = m_pLabelledImageIpl->width;
  inputSize.height = m_pLabelledImageIpl->height;

  m_pTempImage   = cvCreateImage(inputSize, IPL_DEPTH_8U, 1); 
  m_pBoxImageIpl = cvCreateImage(inputSize, IPL_DEPTH_8U, 3); 

  Outline(m_pLabelledImageIpl, m_pTempImage);
  cvCvtColor(m_pTempImage, m_pBoxImageIpl, CV_GRAY2RGB);

  // Draw the boxes

  DrawMinBoxes(m_pBoxImageIpl, m_MinBlobBoxes, 1); 
}

bool SWTExtractor::ValidText(const char *charString)
{
  int stringLength;
  int alphaNumCount;
  int charIndex;
  bool validText;

  stringLength = strlen(charString);
  alphaNumCount = 0;

  for (charIndex = 0; charIndex < stringLength; charIndex++)
  {
    if (isalnum(charString[charIndex]))
      alphaNumCount++;
  }

  if (alphaNumCount > 0.5 * stringLength)
    validText = true;
  else
    validText = false;

  return validText;
}

void SWTExtractor::InitText()
{
  cvInitFont(&m_TextFont, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0);
  m_TextColor = cvScalar(0, 0, 255);
}

void SWTExtractor::DrawText(string textString,
                            RotatedRect minBox)
{
  string displayString;
  CvPoint textLoc;
  int delta;

  // Write recognized text next to blob

//  displayString = textString.substr(0, textString.length() - 2);
  displayString = textString;

  textLoc.x = (int) floor(minBox.center.x + 0.5);
  textLoc.y = (int) floor(minBox.center.y + 0.5);

  delta = minBox.size.width;

  if (minBox.size.height < delta)
    delta = minBox.size.height;

  textLoc.y += delta;

  cvPutText(m_pBoxImageIpl, 
            displayString.c_str(),
            textLoc,
            &m_TextFont,
            m_TextColor);
}

void SWTExtractor::DisplayBoxImage()
{
  // Display
  
  cvNamedWindow("SWTExtractor: box image");
  cvShowImage("SWTExtractor: box image", m_pBoxImageIpl);
  cvWaitKey(0);

  // Cleanup

  cvReleaseImage(&m_pTempImage);
  cvReleaseImage(&m_pBoxImageIpl);
}
