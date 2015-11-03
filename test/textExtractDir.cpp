#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <cstdio>
#include <vector>
#include "boost/filesystem.hpp"

#include "swt.hpp"
#include "json_writer.hpp"
#include "tesseract_textextractor.hpp"
#include "Timer.h"

#define __GLOBAL_OBJECTS
#define LOG_FILE "textExtractDir.log"

using namespace cv;
using namespace gnim;
using namespace papyrus;
using namespace boost::filesystem;
  
#ifdef __GLOBAL_OBJECTS
SWTLabeller  labeller;
SWTBoxFinder boxFinder;
SWTExtractor textExtractor;
#endif

JSONwriter *pJSONWriter;

string inputDir;
string inputList;
string maskDir;
string textDir;

ofstream logFile;

void ImageToText(string imageFilename);

void ProcessImage(Image &inputImage,
                  string maskRootName,
                  string textFilename,
                  vector<image_text_t> &outputText);

// Main

int main(int argc, char** argv)
{
  string inputImageFilename;
  string inputImagePath;
  string jsonFile;

  // Process command line

  if (argc < 5)
  {
    printf("usage: %s <input_dir> <input_list> <mask_dir> <text_dir> (<json_file>)\n", argv[0]);
    return -1;
  }

  inputDir  = string(argv[1]);
  inputList = string(argv[2]);
  maskDir   = string(argv[3]);
  textDir   = string(argv[4]);

  pJSONWriter = NULL;

  if (argc > 5)
  {
    jsonFile = string(argv[5]);
    pJSONWriter = new JSONwriter(jsonFile);
  }

  logFile.open(LOG_FILE, std::ofstream::out | std::ofstream::app);

  // Loop through the images

  ifstream listFile(inputList);

  while (getline(listFile, inputImageFilename))
  {
    inputImagePath = inputDir + "/" + inputImageFilename;

    std::cout << inputImageFilename << endl;

    ImageToText(inputImagePath);
  }

  // Cleanup

  logFile.close();

  if (pJSONWriter)
    delete pJSONWriter;

  exit(0);
}

void ImageToText(string imageFilename)
{
  string maskRootName;
  string textFilename;
  IplImage *pInputImageIpl;
  vector<image_text_t> outputText;

  outputText.clear();

  // Read image

  Image inputImage = gnim::Image::load(imageFilename);

  // Display it

  GNIMToIpl(&inputImage, &pInputImageIpl);

  // Convert to mono

  Image monoImage = inputImage.convert(GNIM_PIX_GRAYSCALE8);

  // Display it

  GNIMToIpl(&monoImage, &pInputImageIpl);

  cvNamedWindow("Input Image", CV_WINDOW_AUTOSIZE);
  cvShowImage("Input Image", pInputImageIpl);
  cvWaitKey(0);

  // Invert pixel values

  Image *pInvertedImage;

  InvertGNIM(&monoImage, &pInvertedImage);

  // Process the original image

  std::cout << "original image" << endl;

  maskRootName = maskDir + "/" + basename(imageFilename) + "_orig_mask";
  textFilename = textDir + "/" + basename(imageFilename) + "_orig_text.txt";

  ProcessImage(monoImage,
               maskRootName,
               textFilename,
               outputText);

  // Process the inverted image

  std::cout << "inverted image" << endl;

  maskRootName = maskDir + "/" + basename(imageFilename) + "_inv_mask";
  textFilename = textDir + "/" + basename(imageFilename) + "_inv_text.txt";

  ProcessImage(*pInvertedImage,
               maskRootName,
               textFilename,
               outputText);

  // JSON output

  if (pJSONWriter)
  {
    pJSONWriter->addImage(imageFilename,
                          outputText.begin(),
                          outputText.end());
  }

  // Cleanup

  cvReleaseImage(&pInputImageIpl);
  delete pInvertedImage;
}

void ProcessImage(Image &inputImage,
                  string maskRootName,
                  string textFilename,
                  vector<image_text_t> &outputText)
{
  int i;
  int j;
  int textStartIndex;
  vector<bounding_box_t> wordBoxes;
  ofstream textFile(textFilename);
  double procTime;
  Timer timer;

#ifndef __GLOBAL_OBJECTS
  SWTLabeller  labeller;
  SWTBoxFinder boxFinder;
  SWTExtractor textExtractor;
#endif

  // Setup

  wordBoxes.clear();

  // Create labelled image

  timer.Start();

  Image labelledImage = labeller(inputImage);

  procTime = timer.Stop();
  logFile << "labeller time = " << procTime << endl;

  // Find boxes

  timer.Start();

  boxFinder(labelledImage, wordBoxes);

  procTime = timer.Stop();
  logFile << "box finder time = " << procTime << endl;

  // Set mask root name

  textExtractor.SetMaskRootName(maskRootName);

  // Extract text

  textStartIndex = outputText.size();
  timer.Start();

  textExtractor(labelledImage,
                wordBoxes.begin(),
                wordBoxes.end(),
                outputText);

  procTime = timer.Stop();
  logFile << "extractor time = " << procTime << endl;

  // Print/save the results 

  for (i = textStartIndex; i < outputText.size(); i++)
  {
    j = i - textStartIndex;
    textFile << "text[" << j << "] = " << outputText[i].second << std::endl;
  }
}
