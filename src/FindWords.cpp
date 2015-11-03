#include <iostream>
#include <string>
#include <utility>
#include <cstdio>
#include <sstream>

#include "FindWords.hpp"

using namespace std;
using namespace cv;

#define FC_PIXEL_DENSITY 0.1
#define FC_REFINE_WORD_ANGLE
// #define FC_SAVE_IMAGES
// #define FC_DISPLAY_IMAGES
#define FC_IMAGE_DIR "./Images/"
// #define FC_PRINT_RESULTS
#define FC_WRAP_ANGLE_180

// Please note ...
//
// There is an inconsistency in OpenCV.  The functions to do rotations of an image assume that a positive angle
// indicates a counter-clockwise rotation.  However, for a RotatedRect structure, the sense of a positive angle
// is clockwise.  If we mix these, such as using the angle from a RotatedRect to control a rotation, we have to
// invert the angle.  Working solely within one representation, there is no inconsistency.
//
// Furthermore, given the angles in RotatedRect arrays and other estimated angle per word, we must make sure
// that we use the correct input angle for any operation.
//
// And another thing ... In OpenCV, the minAreaRect function produces curious output.  The angle in the RotatedRect
// it returns has a range of [-90, 0), including -90 degrees, but not 0.  The angle is -90 whenever the rectangle is
// aligned with the coordinate axes.  The angle increases from -90 to 0 as the rectangle rotates in a clockwise
// direction.  The dimension of the rectangle along the direction of the angle is given as the width.  Note that,
//  since the sense of the rotation angle is clockwise from the positive x-axis, -90 degrees is up, 

bool sort_by_y(Point2f pointA, Point2f pointB);
bool sort_loc_by_x(CharLoc pointA, CharLoc pointB);

FindWords::FindWords()
{
}

FindWords::~FindWords()
{
}

void FindWords::process(IplImage *pSrcImage,
                        int lightDarkText,
                        IplImage *pLabelledWordImage,
                        vector <Rect> &outWordBoxes)
{
  int    i;
  int    wordIndex;
  CvSize imageSize;

  // Create the other images

  imageSize.width  = pSrcImage->width;
  imageSize.height = pSrcImage->height;

  pCharMaskImage = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
  pWordMaskImage = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);

  // Create the TextDetect objects

  Init(lightDarkText);

  // The main function calls

  pCharDetect->GetText(pSrcImage, charMasks, charBoxes);
  pWordDetect->GetText(pSrcImage, wordMasks, wordBoxes);

  // Estimate the chars per word

  CharBlobsInWordBlobs();

  // Create the labelled word image

  LabelWords(pLabelledWordImage);

  // Find min boxes

  // For words

  FindMinBoxes(wordMasks, wordBoxes, wordMinBoxes);

  // For characters

  FindMinBoxes(charMasks, charBoxes, charMinBoxes);

  // Get the mask images
 
  pCharDetect->GetImage(lightDarkText, "MASK", pCharMaskImage);
  pWordDetect->GetImage(lightDarkText, "MASK", pWordMaskImage);

#ifdef FC_DISPLAY_IMAGES
  // Display full images

  DisplayImages();
#endif

  // Output an image of labelled work blobs
  // and the bounding box for each word
  // Do this for words that actually have (enough) candidate characters

  outWordBoxes.clear();

  for (wordIndex = 0; wordIndex < wordBoxes.size(); wordIndex++)
  {
#if 0
    if (charImages[wordIndex].size() < 2)
      continue;
#endif

    outWordBoxes.push_back(wordBoxes[wordIndex]);
  }

#if 0
  // Print results

  printf("total characters = %d\n", charBoxes.size());
  printf("total words = %d\n", wordBoxes.size());

  for (wordIndex = 0; wordIndex < wordBoxes.size(); wordIndex++)
  {
    printf("word %d has %d chars\n", wordIndex, charsInWords[wordIndex].size());
  }
#endif

  // Cleanup

  cvReleaseImage(&pCharMaskImage);
  cvReleaseImage(&pWordMaskImage);
  delete pWordDetect;
  delete pCharDetect;
}

void FindWords::Init(int lightDarkText)
{
  // Clear vectors and images

  wordBoxes.clear();
  wordMinBoxes.clear();
  wordLowBoxes.clear();
  wordMasks.clear();
  wordAngles.clear();
  charBoxes.clear();
  charMinBoxes.clear();
  charMasks.clear();
  charsInWords.clear();
  charImages.clear();

  cvZero(pCharMaskImage);
  cvZero(pWordMaskImage);

  // Create the TextDetect objects
  
  SetCharConfig();
  SetWordConfig();

  if (lightDarkText == 0)
  {
    charConfig.lightText = true;
    wordConfig.lightText = true;
  }
  else
  {
    charConfig.darkText = true;
    wordConfig.darkText = true;
  }

  pCharDetect = new TextDetect(&charConfig);
  pWordDetect = new TextDetect(&wordConfig);
}

void FindWords::SetWordConfig()
{
  // Modes

  wordConfig.darkText      = false;
  wordConfig.lightText     = false;
  wordConfig.logSWT        = false;
  wordConfig.smoothSWT     = false;
  wordConfig.separateBlobs = true;
  wordConfig.mergeBlobs    = true;
  wordConfig.suppressEdges = false;
  wordConfig.padRegions    = false;
  wordConfig.suppressSmallRegions = true;
  wordConfig.suppressThickStrokes = true;
  wordConfig.displayImages = true;
  wordConfig.saveImage     = false;

  // Parameters

  wordConfig.maxNumBlobs = 10000;
  wordConfig.angleRangeDivisor = 2;
  wordConfig.minStrokeWidthDivisor = 20;
  wordConfig.maxStrokeWidthDivisor = 10;
  wordConfig.smoothFilterSize = 3;
  wordConfig.separateFilterSize  = 1;
  wordConfig.mergeFilterSize  = 5;
  wordConfig.dilateRadius     = 0;
  wordConfig.padWidth         = 0;
  wordConfig.cannyLowThreshold  = 200;
  wordConfig.cannyHighThreshold = 200;  // was 250
  wordConfig.fgMinValue      = 100;
  wordConfig.strokePixelDiff = 5;
}
 
void FindWords::SetCharConfig()
{
  // Modes

  charConfig.darkText      = false;
  charConfig.lightText     = false;
  charConfig.logSWT        = false;
  charConfig.smoothSWT     = false;
  charConfig.separateBlobs = true;  // was true
  charConfig.mergeBlobs    = false;
  charConfig.suppressEdges = false;
  charConfig.padRegions    = false;
  charConfig.suppressSmallRegions = true;
  charConfig.suppressThickStrokes = false;
  charConfig.displayImages = true;

  // Parameters

  charConfig.maxNumBlobs = 10000;
  charConfig.angleRangeDivisor = 2;
  charConfig.minStrokeWidthDivisor = 50;
  charConfig.maxStrokeWidthDivisor = 10;
  charConfig.smoothFilterSize = 3;
  charConfig.separateFilterSize = 1;
  charConfig.mergeFilterSize  = 1;
  charConfig.dilateRadius     = 0;
  charConfig.padWidth         = 0;
  charConfig.cannyLowThreshold  = 200;
  charConfig.cannyHighThreshold = 200;  // was 250
  charConfig.fgMinValue      = 100;
  charConfig.strokePixelDiff = 5;
}

void FindWords::CharBlobsInWordBlobs()
{
  vector<int> charsInCurrentWord;
  int wordIndex;
  int charIndex;
  bool charInWord;

  charsInWords.clear();

  for (wordIndex = 0; wordIndex < wordMasks.size(); wordIndex++)
  {
    charsInCurrentWord.clear();

    for (charIndex = 0; charIndex < charMasks.size(); charIndex++)
    {
      charInWord = IsCharInWord(charMasks[charIndex],
                                charBoxes[charIndex],
                                wordMasks[wordIndex],
                                wordBoxes[wordIndex]);

      if (charInWord)
        charsInCurrentWord.push_back(charIndex);
    }

    charsInWords.push_back(charsInCurrentWord);
  }
}

bool FindWords::IsCharInWord(Mat &charMask, Rect &charRect, Mat &wordMask, Rect &wordRect)
{
  bool retCode;
  Rect intersectRect;
  Rect charRoi;
  Rect wordRoi;
  Mat  charRoiImg;
  Mat  wordRoiImg;
  Mat  andRoiImg;
  int numPixelsOverlap;

  retCode = DoRectsOverlap(charRect, wordRect);

  if (!retCode)
    return false;

  intersectRect = charRect & wordRect; 

  charRoi = intersectRect;
  charRoi.x -= charRect.x;
  charRoi.y -= charRect.y;

  wordRoi = intersectRect;
  wordRoi.x -= wordRect.x;
  wordRoi.y -= wordRect.y;

  charRoiImg = Mat(charMask, charRoi);
  wordRoiImg = Mat(wordMask, wordRoi);

  andRoiImg = charRoiImg.clone();

  bitwise_and(charRoiImg, wordRoiImg, andRoiImg);

  numPixelsOverlap = countNonZero(andRoiImg);

  if ((float) numPixelsOverlap > FC_PIXEL_DENSITY * (andRoiImg.rows * andRoiImg.cols))
    return true;
  else
    return false;
}

void FindWords::LabelWords(IplImage *pLabelledWordImage)
{
  int wordIndex;
  int charIndex;
  int charCount;
  float wordLabel;
  int numCharsInWord;
  IplImage tempImage;
  IplImage *pTempImage;
  IplImage *pMaskImage;
  IplImage *pPrevMaskImage;
  CvRect charROI;
  CvSize imageSize;
  int maskArea;

  imageSize.width  = pLabelledWordImage->width;
  imageSize.height = pLabelledWordImage->height;

  pMaskImage       = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
  pPrevMaskImage   = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);

  cvZero(pLabelledWordImage);

  // Loop through words
  // Note that we only label "word" blobs that contain character
  // blobs

  wordLabel = 0;

  for (wordIndex = 0; wordIndex < wordMasks.size(); wordIndex++)
  {
    // Check number of characters
    // If zero, skip this "word" blob"

    numCharsInWord = charsInWords[wordIndex].size();

    if (numCharsInWord == 0)
    {
      // `printf("word %d has no characters\n", wordIndex);
      continue;
    }

    // Setup per word

    cvZero(pMaskImage);

    // Loop through characters in a word
    // Composit the character masks to create  word mask
    // that is 255 within the word and 0 outside

    for (charCount = 0; charCount < numCharsInWord; charCount++)
    {
      charIndex  = charsInWords[wordIndex][charCount];
      tempImage  = IplImage(charMasks[charIndex]);
      pTempImage = &tempImage;

      charROI    = CvRect(charBoxes[charIndex]);
      cvSetImageROI(pMaskImage, charROI);

      cvOr(pMaskImage, pTempImage, pMaskImage);
    }

    cvResetImageROI(pMaskImage);

    // Are there pixels in the mask?  If not, continue

    maskArea = cvCountNonZero(pMaskImage);

    if (maskArea == 0)
    {
      // printf("word %d has no pixels\n", wordIndex);
      continue;
    }

    // Are there NEW pixels in the mask; do not disturb old words

    cvCmpS(pLabelledWordImage, 0.5, pPrevMaskImage, CMP_GT);

    cvNot(pPrevMaskImage, pPrevMaskImage);

    cvAnd(pMaskImage, pPrevMaskImage, pMaskImage);

    maskArea = cvCountNonZero(pMaskImage);

    if (maskArea == 0)
    {
      // printf("word %d has no new pixels\n", wordIndex);
      continue;
    }

    // Go ahead, label new pixels in this word
    
    wordLabel++;

    // Combine with other labelled words; at a pixel,
    // the latest word wins

#if 0
    cvXor(pLabelledWordImage,
          pLabelledWordImage,
          pLabelledWordImage,
          pMaskImage);
#endif

    cvAddS(pLabelledWordImage,
           Scalar(wordLabel),
           pLabelledWordImage,
           pMaskImage);
  }

  cvResetImageROI(pMaskImage);
  cvReleaseImage(&pMaskImage);
  cvReleaseImage(&pPrevMaskImage);
}

void FindWords::GetWordAngles()
{
  int wordIndex;
  double estWordAngle;

  wordAngles.clear();
  wordLowBoxes.clear();

  for (wordIndex = 0; wordIndex < wordMasks.size(); wordIndex++)
  {
#ifdef FC_REFINE_WORD_ANGLE
    if (charsInWords[wordIndex].size() > 1)
      getWordAngle(wordIndex, estWordAngle);
    else
      estWordAngle = ConvertAngle(wordMinBoxes[wordIndex]);
#else
    estWordAngle = ConvertAngle(wordMinBoxes[wordIndex]);
#endif

    wordAngles.push_back(estWordAngle);
  }
}

void FindWords::getWordAngle(int wordIndex, double &estWordAngle)
{
  int charIndex;
  Point2f wordCenter;
  double wordAngle;
  vector<int> charIndices;
  vector<RotatedRect> charMinBoxesInWord;
  Point2f corners[4];
  vector<Point2f> lowPoints;
  RotatedRect lowBox;

  // Get indices of characters in this word

  charIndices = charsInWords[wordIndex];

  // Get word center and angle from its bounding box

  wordCenter = wordMinBoxes[wordIndex].center;
  wordAngle  = ConvertAngle(wordMinBoxes[wordIndex]);

  // Get the min bounding boxes for the characters

  charMinBoxesInWord.clear();

  for (charIndex = 0; charIndex < charIndices.size(); charIndex++)
  {
    charMinBoxesInWord.push_back(charMinBoxes[charIndices[charIndex]]);
  }

  // Normalize the character boxes relative to the word box

  for (charIndex = 0; charIndex < charIndices.size(); charIndex++)
  {
    charMinBoxesInWord[charIndex].center -= wordCenter;
    charMinBoxesInWord[charIndex].angle  -= wordAngle;
  }

  // For each character box, compute its corners, sort them, and push the
  // bottom two onto a vector of points.

  lowPoints.clear();

  for (charIndex = 0; charIndex < charIndices.size(); charIndex++)
  {
    charMinBoxesInWord[charIndex].points(corners);

    vector<Point2f> cornerVec(corners, corners + 4);

    sort(cornerVec.begin(), cornerVec.end(), sort_by_y);

    lowPoints.push_back(cornerVec[2]);
    lowPoints.push_back(cornerVec[3]);
  }

  // Get the min bounding box of this vector of low points

  lowBox = minAreaRect(lowPoints);

  // Return the angle from this box

  estWordAngle = ConvertAngle(lowBox) + wordAngle;

#ifdef FC_WRAP_ANGLE_180
  // Both angles range from  -90 to + 90, so we wrap their
  // sum into the same range

  if (estWordAngle > 90)
    estWordAngle -= 180;
  if (estWordAngle < -90)
    estWordAngle += 180;
#endif

  // Adjust the min bounding  box and push it onto a vector

  lowBox.angle  += wordAngle;
  lowBox.center += wordCenter;

  wordLowBoxes.push_back(lowBox);

}

double FindWords::ConvertAngle(RotatedRect &minBox)
{
  double inputAngle;
  double outputAngle;

  inputAngle = minBox.angle;

  if (minBox.size.height > minBox.size.width)
    outputAngle = inputAngle + 90;  // 0 to 90
  else 
    outputAngle = inputAngle;       // -90 to 0 - we leave it

  return outputAngle;
}

void FindWords::SortCharsInWord(int wordIndex)
{
  int charIndex;
  CharLoc currCharLoc;
  vector<CharLoc> charLocations;
  vector<CharLoc> alignedLocations;

  Point2f wordCenter;
  double wordAngle;

  // Return for trivial cases
  // (nothing to be done)

  if (charsInWords[wordIndex].size() < 2)
    return;

  // Otherwise

  charLocations.clear();

  for (charIndex = 0; charIndex < charsInWords[wordIndex].size(); charIndex++)
  {
    currCharLoc.charIndex = charsInWords[wordIndex][charIndex];
    currCharLoc.center    = charMinBoxes[currCharLoc.charIndex].center;

    charLocations.push_back(currCharLoc);
  }

  wordCenter = wordMinBoxes[wordIndex].center;
  wordAngle  = wordAngles[wordIndex];  // The estimated angle, not just from the word bounding box

  AlignLocations(charLocations,
                 alignedLocations,
                 wordCenter,
                 wordAngle);

  sort(alignedLocations, sort_loc_by_x);

  // Now we remove the unsorted indices from charsInWords[wordIndex] and
  // replace them by the sorted indices

  charsInWords[wordIndex].clear();

  for (charIndex = 0; charIndex < alignedLocations.size(); charIndex++)
  {
    charsInWords[wordIndex].push_back(alignedLocations[charIndex].charIndex);
  }
}

void FindWords::AlignLocations(vector<CharLoc> &inputLocs,
                               vector<CharLoc> &outputLocs,
                               Point2f wordCenter,
                               double wordAngle)
{
  int i;
  int numInPoints;
  CharLoc tempPoint;
  double x;
  double y;
  double sinAngle;
  double cosAngle;

  // Setup

  outputLocs.clear();
  numInPoints = inputLocs.size();

  sinAngle = sin(wordAngle);
  cosAngle = cos(wordAngle);

  // Loop through the input points

  for (i = 0; i < numInPoints; i++)
  {
    // Subtract the input center

    x = inputLocs[i].center.x - wordCenter.x;
    y = inputLocs[i].center.y - wordCenter.y;

    // Rotate by -angle
    // Note that increasing y is downwards, and positive rotation is
    // clockwise, so the inverse rotation matrix looks normal.

    tempPoint.center.x = cosAngle * x + sinAngle * y;
    tempPoint.center.y = cosAngle * y - sinAngle * x;

    tempPoint.charIndex = inputLocs[i].charIndex;

    // Push onto output vector

    outputLocs.push_back(tempPoint);
  }
}

void FindWords::AlignCharImages()
{
  vector<Mat> currWordImages;
  Mat *pRotatedImage;
  int wordIndex;
  int charIndex;
  int charImageIndex;

  charImages.clear();
  
  for (wordIndex = 0; wordIndex < wordMasks.size(); wordIndex++)
  {
    currWordImages.clear();

    for (charIndex = 0; charIndex < charsInWords[wordIndex].size(); charIndex++)
    {
      charImageIndex = charsInWords[wordIndex][charIndex];

     // Notes about the rotation angle
     // the sense of the rotation angle is counter-clockwise,
     // but the sense of the measured angle is clockwise
     //
     // Also, because of the logic so far, a horizontal word
     // corresponds to a measured angle of -90, not 0.

      pRotatedImage  = RotateImage(charMasks[charImageIndex],
                                   wordAngles[wordIndex] + 90,
                                   true);

      currWordImages.push_back(*pRotatedImage);
    }

    charImages.push_back(currWordImages);
  }
}

void FindWords::SaveImages(IplImage *pSrcImage)
{
  int wordIndex;
  int charIndex;
  stringstream stringStream;
  string imageFilename;
  Mat wordROI;

  for (wordIndex = 0; wordIndex < charImages.size(); wordIndex++)
  {
    // Word mask

    stringStream.str("");
    stringStream << FC_IMAGE_DIR << "word_mask_" << wordIndex << ".jpg";
    imageFilename = stringStream.str();

    // imwrite(imageFilename, wordMasks[wordIndex]);

    // Word ROI pixels

    stringStream.str("");
    stringStream << FC_IMAGE_DIR << "word_image_" << wordIndex << ".jpg";
    imageFilename = stringStream.str();

    wordROI = Mat(Mat(pSrcImage), wordBoxes[wordIndex]);

    // imwrite(imageFilename, wordROI);
  }

  for (wordIndex = 0; wordIndex < charImages.size(); wordIndex++)
  {
    for (charIndex = 0; charIndex < charImages[wordIndex].size(); charIndex++)
    {
      stringStream.str("");
      stringStream << FC_IMAGE_DIR << "word_" << wordIndex << "_char_" << charIndex << ".jpg";
      imageFilename = stringStream.str();

      // imwrite(imageFilename, charImages[wordIndex][charIndex]);
    }
  }

#ifdef FC_PRINT_RESULTS
// Print initial and updated word angles

  FILE *pOutFile;

  pOutFile = fopen("word_angles.txt", "w");

  for (wordIndex = 0; wordIndex < charImages.size(); wordIndex++)
  {
     fprintf(pOutFile, "%d %f %f %f %f %f %f %d\n",
             wordIndex,
             wordMinBoxes[wordIndex].center.x,
             wordMinBoxes[wordIndex].center.y,
             wordMinBoxes[wordIndex].size.width,
             wordMinBoxes[wordIndex].size.height,
             wordMinBoxes[wordIndex].angle,
             wordAngles[wordIndex],
             charImages[wordIndex].size());    
  }

  fclose(pOutFile);
#endif
}
  
IplImage* FindWords::UprightBoxImage(IplImage *pSrcImage,
                                     vector<Rect> &textBoxes)
{ 
  IplImage *pTextBoxImage;
  CvSize imageSize;
  
  imageSize.width  = pSrcImage->width;
  imageSize.height = pSrcImage->height;
  
  pTextBoxImage = cvCreateImage(imageSize, IPL_DEPTH_8U, 3);
  
  cvCvtColor(pSrcImage, pTextBoxImage, CV_GRAY2RGB);
  DrawBoxes(pTextBoxImage, textBoxes);

  return pTextBoxImage;
}

IplImage* FindWords::MinBoxImage(IplImage *pSrcImage,
                                 vector<RotatedRect> &textBoxes,
                                 vector<RotatedRect> &lowBoxes)
{
  IplImage *pTextBoxImage;
  CvSize imageSize;

  imageSize.width  = pSrcImage->width;
  imageSize.height = pSrcImage->height;

  pTextBoxImage = cvCreateImage(imageSize, IPL_DEPTH_8U, 3);

  cvCvtColor(pSrcImage, pTextBoxImage, CV_GRAY2RGB);
  DrawMinBoxes(pTextBoxImage, textBoxes);

  if (lowBoxes.size() > 0)
    DrawMinBoxes(pTextBoxImage, lowBoxes, 1);

  return pTextBoxImage;
}

void FindWords::DisplayImages()
{
  IplImage *pCharUprightBoxImage;
  IplImage *pWordUprightBoxImage;
  IplImage *pCharMinBoxImage;
  IplImage *pWordMinBoxImage;

  // Box images

  pCharUprightBoxImage = UprightBoxImage(pCharMaskImage, charBoxes);
  pWordUprightBoxImage = UprightBoxImage(pWordMaskImage, wordBoxes);
  pCharMinBoxImage = MinBoxImage(pCharMaskImage, charMinBoxes, wordLowBoxes);
  pWordMinBoxImage = MinBoxImage(pWordMaskImage, wordMinBoxes, wordLowBoxes);

  // Create windows

  cvNamedWindow("Word Mask Image", CV_WINDOW_AUTOSIZE);
  cvNamedWindow("Char Mask Image", CV_WINDOW_AUTOSIZE);
  cvNamedWindow("Word Upright Box Image", CV_WINDOW_AUTOSIZE);
  cvNamedWindow("Char Upright Box Image", CV_WINDOW_AUTOSIZE);
  cvNamedWindow("Word Min Box Image", CV_WINDOW_AUTOSIZE);
  cvNamedWindow("Char Min Box Image", CV_WINDOW_AUTOSIZE);

  // Display them

  cvShowImage("Word Mask Image", pWordMaskImage);
  cvShowImage("Char Mask Image", pCharMaskImage);
  cvShowImage("Word Upright Box Image", pWordUprightBoxImage);
  cvShowImage("Char Upright Box Image", pCharUprightBoxImage);
  cvShowImage("Word Min Box Image", pWordMinBoxImage);
  cvShowImage("Char Min Box Image", pCharMinBoxImage);

  cvWaitKey(0);

  // Cleanup

  cvDestroyAllWindows();
  cvReleaseImage(&pCharUprightBoxImage);
  cvReleaseImage(&pWordUprightBoxImage);
  cvReleaseImage(&pCharMinBoxImage);
  cvReleaseImage(&pWordMinBoxImage);
}

bool sort_by_y(Point2f pointA, Point2f pointB)
{
  if (pointA.y < pointB.y)
    return true;
  else
    return false;
}

bool sort_loc_by_x(CharLoc pointA, CharLoc pointB)
{
  if (pointA.center.x < pointB.center.x)
    return true;
  else
    return false;
}
