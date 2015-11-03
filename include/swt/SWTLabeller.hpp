// SWTLabeller.hpp
//
// Peter Wendt
// Gracenote LLC
// Aug 19, 2015
//

#pragma once

#include "gnim.hpp"
#include "papyrus.hpp"

#include "FindWords.hpp"

namespace papyrus
{
  class SWTLabeller : public TextLabeller
  {
   public:

    SWTLabeller();

    virtual ~SWTLabeller();

    virtual gnim::Image operator()(const gnim::Image &image);

   private:

    FindWords findWords;
  };
}
