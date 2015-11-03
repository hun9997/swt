#include <iostream>
#include <string>
#include <utility>
#include <cstdio>
#include <vector>

#include "swt.hpp"
#include "tesseract_textextractor.hpp"

using namespace cv;
using namespace gnim;
using namespace papyrus;

// Main

int main(int argc, char** argv)
{
  int    i;
  string inputImageFilename;

  vector<image_text_t> outputText;
  
  TextEngine textEngine(unique_ptr<TextLabeller> (new SWTLabeller),
             unique_ptr<TextBoxFinder> (new SWTBoxFinder),
             unique_ptr<TextExtractor> (new TesseractTextExtractor));

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

  // Extract text

  textEngine(monoImage,
             outputText);

  // Print/save the results 

  for (i = 0; i < outputText.size(); i++)
    std::cout << "text[" << i << "] = " << outputText[i].second << std::endl;

  exit(0);
}
