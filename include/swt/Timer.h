//
// Timer.h
// - Timer class: used to time sections of code
//
// Prashant Ramanathan
// August 22, 2008
// Zeitera LLC
//

#include <sys/time.h>

#ifndef _TIMER_H
#define _TIMER_H

class Timer
{
 public:
  Timer();
  ~Timer();
  void Start();
  double Stop();

 private:
  timeval start, end;
};  

#endif // _TIMER_H
