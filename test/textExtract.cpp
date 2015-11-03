#include <iostream>
#include <string>
#include <utility>
#include <cstdio>
#include <vector>

#include "swt.hpp"
#include "tesseract_textextractor.hpp"

#define RESIZED_HEIGHT 512

using namespace cv;
using namespace gnim;
using namespace papyrus;

// Main

int main(int argc, char** argv)
{
  int    i;
  string inputImageFilename;

  vector<bounding_box_t> wordBoxes;
  vector<bounding_box_t> wordBoxes2;
  vector<image_text_t>   outputText;
  vector<image_text_t>   outputText2;
  
  SWTLabeller  labeller;
  SWTBoxFinder boxFinder;
  SWTExtractor textExtractor;

  // Process command line

  if (argc < 2)
  {
    printf("usage: %s <input_image>\n", argv[0]);
    return -1;
  }

  inputImageFilename = string(argv[1]);

  // Read image

  Image inputImage = gnim::Image::load(inputImageFilename);

  printf("width = %d\n", inputImage.width());
  printf("height = %d\n", inputImage.height());
  printf("num channels = %d\n", inputImage.depth());

  // Convert to mono

  Image monoImage = inputImage.convert(GNIM_PIX_GRAYSCALE8);

  // Invert pixel values

  Image *pInvertedImage;

  InvertGNIM(&monoImage, &pInvertedImage);

  // Process the original image

  std::cout << "original image" << endl;

#if 1

  // Create labelled image

  Image labelledImage = labeller(monoImage);

  // Find boxes

  boxFinder(labelledImage, wordBoxes);

  // Extract text

  textExtractor.SetMaskRootName("mask_orig");

  textExtractor(labelledImage,
                wordBoxes.begin(),
                wordBoxes.end(),
                outputText);

  // Print/save the results 

  for (i = 0; i < outputText.size(); i++)
    std::cout << "text[" << i << "] = " << outputText[i].second << std::endl;

#endif

  // Process the inverted image

  std::cout << "inverted image" << endl;

  // Create labelled image

  Image labelledImage2 = labeller(*pInvertedImage);

  // Find boxes

  boxFinder(labelledImage2, wordBoxes2);

  // Extract text

  textExtractor.SetMaskRootName("mask_inv");

  textExtractor(labelledImage2,
                wordBoxes2.begin(),
                wordBoxes2.end(),
                outputText2);

  // Print/save the results 

  for (i = 0; i < outputText2.size(); i++)
    std::cout << "text[" << i << "] = " << outputText2[i].second << std::endl;

  // Cleanup

  delete pInvertedImage;

  exit(0);
}
