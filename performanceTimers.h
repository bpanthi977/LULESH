/// \file
/// Performance timer functions.
#ifndef __PERFORMANCE_TIMERS_H_
#define __PERFORMANCE_TIMERS_H_

#include <stdio.h>

/// Timer handles
enum TimerHandle{
   TH_loop,
   TH_timestep,
   TH_lagrangeNodal_approx,
   TH_ln_aprx_prep,
   TH_lagrangeNodal,
   TH_calcForceForNodes,
   TH_calcAccForNodes,
   TH_applyAccBC,
   TH_calcVelForNodes,
   TH_calcPosForNodes,
   TH_lagrangeElements,
   TH_calcLagrangeElements,
   TH_calcQForElems,
   TH_applyMaterialProp,
   TH_updateVolForElems,
   TH_calcTimeConstraintsForElems,
   numberOfTimers
};

/// Use the startTimer and stopTimer macros for timers in code regions
/// that may be performance sensitive.  These can be compiled away by
/// defining NTIMING.  If you are placing a timer anywere outside of a
/// tight loop, consider calling profileStart and profileStop instead.
///
/// Place calls as follows to collect time for code pieces.
/// Time is collected everytime this portion of code is executed.
///
///     ...
///     startTimer(computeForceTimer);
///     computeForce(sim);
///     stopTimer(computeForceTimer);
///     ...
///
#ifndef NTIMING
#define startTimer(handle)    \
   do                         \
{                          \
   profileStart(handle);   \
} while(0)
#define stopTimer(handle)     \
   do                         \
{                          \
   profileStop(handle);    \
} while(0)
#else
#define startTimer(handle)
#define stopTimer(handle)
#endif

/// Use profileStart and profileStop only for timers that should *never*
/// be turned off.  Typically this means they are outside the main
/// simulation loop.  If the timer is inside the main loop use
/// startTimer and stopTimer instead.
void profileStart(const enum TimerHandle handle);
void profileStop(const enum TimerHandle handle);

/// Use to get elapsed time (lap timer).
double getElapsedTime(const enum TimerHandle handle);

/// Print timing results.
void printPerformanceResults();

/// Print timing results to Yaml file
void printPerformanceResultsYaml(FILE* file);
#endif
