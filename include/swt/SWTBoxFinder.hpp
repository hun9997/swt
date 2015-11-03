// SWTBoxFinder.hpp
//
// Peter Wendt
// Gracenote LLC
// Aug. 19, 2015
//

#pragma once

#include "gnim.hpp"
#include "papyrus.hpp"

namespace papyrus
{
  class SWTBoxFinder : public TextBoxFinder
  {
   public:

    SWTBoxFinder();

    virtual ~SWTBoxFinder();

    virtual void operator()(const gnim::Image& labelled_image,
                            papyrus::inserter<papyrus::bounding_box_t> output);

   private:
  };
}
