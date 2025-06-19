/// \file
/// Performance timer functions.
///
/// Use the timer functionality to collect timing and number of calls
/// information for chosen computations (such as force calls) and
/// communication (such as sends, receives, reductions).  Timing results
/// are reported at the end of the run showing overall timings and
/// statistics of timings across ranks.
///
/// A new timer can be added as follows:
/// -# add new handle to the TimerHandle in performanceTimers.h
/// -# provide a corresponding name in timerName
///
/// Note that the order of the handles and names must be the
/// same. This order also determines the order in which the timers are
/// printed. Names can contain leading spaces to show a hierarchical
/// ordering.  Timers with zero calls are omitted from the report.
///
/// Raw timer data is obtained from the getTime() and getTick()
/// functions.  The supplied portable versions of these functions can be
/// replaced with platform specific versions for improved accuracy or
/// lower latency.
/// \see TimerHandle
/// \see getTime
/// \see getTick
///


#include "performanceTimers.h"

#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

#include "performanceTimers.h"

int getMyRank()
{
  return 0;
}

int getNRanks() {
  return 1;
}

int printRank() {
  return (getMyRank() == 0);
}


typedef struct RankReduceDataSt
{
   double val;
   int rank;
} RankReduceData;

void addDoubleParallel(double* sendBuf, double* recvBuf, int count)
{
#if USE_MPI
   MPI_Allreduce(sendBuf, recvBuf, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
   for (int ii=0; ii<count; ++ii)
      recvBuf[ii] = sendBuf[ii];
#endif
}

void minRankDoubleParallel(RankReduceData* sendBuf, RankReduceData* recvBuf, int count)
{
#if USE_MPI
   MPI_Allreduce(sendBuf, recvBuf, count, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
#else
   for (int ii=0; ii<count; ++ii)
   {
      recvBuf[ii].val = sendBuf[ii].val;
      recvBuf[ii].rank = sendBuf[ii].rank;
   }
#endif
}

void maxRankDoubleParallel(RankReduceData* sendBuf, RankReduceData* recvBuf, int count)
{
#if USE_MPI
   MPI_Allreduce(sendBuf, recvBuf, count, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
#else
   for (int ii=0; ii<count; ++ii)
   {
      recvBuf[ii].val = sendBuf[ii].val;
      recvBuf[ii].rank = sendBuf[ii].rank;
   }
#endif
}

static uint64_t getTime(void);
static double getTick(void);
static void timerStats(void);

/// You must add timer name in same order as enum in .h file.
/// Leading spaces can be specified to show a hierarchy of timers.
const char* timerName[numberOfTimers] = {
  "total,",
  "timestep",
  "  lagrangeNodal",
  "    calcForceForNodes",
  "       calcElemSFD",
  "       ..calcFBHGF4Elems",
  "    calcAccForNodes",
  "    applyAccBC",
  "    calcVelForNodes",
  "    calcPosForNodes",
  "  lagrangeElements",
  "    calcLagrangeElems",
  "    calcQForElems",
  "    applyMaterialProp",
  "    updateVolForElems",
  "  calcTimeConstrForElems",
};

/// Timer data collected.  Also facilitates computing averages and
/// statistics.
typedef struct TimersSt
{
   uint64_t start;     //!< call start time
   uint64_t total;     //!< current total time
   uint64_t count;     //!< current call count
   uint64_t elapsed;   //!< lap time

   int minRank;        //!< rank with min value
   int maxRank;        //!< rank with max value

   double minValue;    //!< min over ranks
   double maxValue;    //!< max over ranks
   double average;     //!< average over ranks
   double stdev;       //!< stdev across ranks
} Timers;

/// Global timing data collected.
typedef struct TimerGlobalSt
{
   double atomRate;       //!< average time (us) per atom per rank
   double atomAllRate;    //!< average time (us) per atom
   double atomsPerUSec;   //!< average atoms per time (us)
} TimerGlobal;

static Timers perfTimer[numberOfTimers];

void profileStart(const enum TimerHandle handle)
{
   perfTimer[handle].start = getTime();
}

void profileStop(const enum TimerHandle handle)
{
   perfTimer[handle].count += 1;
   uint64_t delta = getTime() - perfTimer[handle].start;
   perfTimer[handle].total += delta;
   perfTimer[handle].elapsed += delta;
}

/// \details
/// Return elapsed time (in seconds) since last call with this handle
/// and clear for next lap.
double getElapsedTime(const enum TimerHandle handle)
{
   double etime = getTick() * (double)perfTimer[handle].elapsed;
   perfTimer[handle].elapsed = 0;

   return etime;
}

/// \details
/// The report contains two blocks.  The upper block is performance
/// information for the printRank.  The lower block is statistical
/// information over all ranks.
void printPerformanceResults()
{
   // Collect timer statistics overall and across ranks
   timerStats();

   if (!printRank())
      return;

   // only print timers with non-zero values.
   double tick = getTick();
   double loopTime = perfTimer[TH_loop].total*tick;

   fprintf(stdout, "                Timer        # Calls    Avg/Call (s)   Total (s)    %% Loop\n");
   fprintf(stdout, "___________________________________________________________________\n");
   for (int ii=0; ii<numberOfTimers; ++ii)
   {
      double totalTime = perfTimer[ii].total*tick;
      if (perfTimer[ii].count > 0)
	 fprintf(stdout, "%-24s%12lu     %8.4f      %8.4f    %8.2f\n",
		 timerName[ii],
		 perfTimer[ii].count,
		 totalTime/(double)perfTimer[ii].count,
		 totalTime,
		 totalTime/loopTime*100.0);
   }

   fprintf(stdout, "\nTiming Statistics Across %d Ranks:\n", getNRanks());
   fprintf(stdout, "                Timer        Rank: Min(s)       Rank: Max(s)      Avg(s)    Stdev(s)\n");
   fprintf(stdout, "_____________________________________________________________________________\n");

   for (int ii = 0; ii < numberOfTimers; ++ii)
   {
      if (perfTimer[ii].count > 0)
	 fprintf(stdout, "%-24s%6d:%10.4f  %6d:%10.4f  %10.4f  %10.4f\n",
	    timerName[ii],
	    perfTimer[ii].minRank, perfTimer[ii].minValue*tick,
	    perfTimer[ii].maxRank, perfTimer[ii].maxValue*tick,
	    perfTimer[ii].average*tick, perfTimer[ii].stdev*tick);
   }
}

/// Returns current time as a 64-bit integer.  This portable version
/// returns the number of microseconds since mindight, Jamuary 1, 1970.
/// Hence, timing data will have a resolution of 1 microsecond.
/// Platforms with access to calls with lower latency or higher
/// resolution (such as a cycle counter) may wish to replace this
/// implementation and change the conversion factor in getTick as
/// appropriate.
/// \see getTick for the conversion factor between the integer time
/// units of this function and seconds.
static uint64_t getTime(void)
{
   struct timeval ptime;
   uint64_t t = 0;
   gettimeofday(&ptime, (struct timezone *)NULL);
   t = ((uint64_t)1000000)*(uint64_t)ptime.tv_sec + (uint64_t)ptime.tv_usec;

   return t;
}

/// Returns the factor for converting the integer time reported by
/// getTime into seconds.  The portable getTime returns values in units
/// of microseconds so the conversion is simply 1e-6.
/// \see getTime
static double getTick(void)
{
   double seconds_per_cycle = 1.0e-6;
   return seconds_per_cycle;
}

/// Collect timer statistics across ranks.
void timerStats(void)
{
   double sendBuf[numberOfTimers], recvBuf[numberOfTimers];

   // Determine average of each timer across ranks
   for (int ii = 0; ii < numberOfTimers; ii++)
      sendBuf[ii] = (double)perfTimer[ii].total;
   addDoubleParallel(sendBuf, recvBuf, numberOfTimers);

   for (int ii = 0; ii < numberOfTimers; ii++)
      perfTimer[ii].average = recvBuf[ii] / (double)getNRanks();


   // Determine min and max across ranks and which rank
   RankReduceData reduceSendBuf[numberOfTimers], reduceRecvBuf[numberOfTimers];
   for (int ii = 0; ii < numberOfTimers; ii++)
   {
      reduceSendBuf[ii].val = (double)perfTimer[ii].total;
      reduceSendBuf[ii].rank = getMyRank();
   }
   minRankDoubleParallel(reduceSendBuf, reduceRecvBuf, numberOfTimers);
   for (int ii = 0; ii < numberOfTimers; ii++)
   {
      perfTimer[ii].minValue = reduceRecvBuf[ii].val;
      perfTimer[ii].minRank = reduceRecvBuf[ii].rank;
   }
   maxRankDoubleParallel(reduceSendBuf, reduceRecvBuf, numberOfTimers);
   for (int ii = 0; ii < numberOfTimers; ii++)
   {
      perfTimer[ii].maxValue = reduceRecvBuf[ii].val;
      perfTimer[ii].maxRank = reduceRecvBuf[ii].rank;
   }

   // Determine standard deviation
   for (int ii = 0; ii < numberOfTimers; ii++)
   {
      double temp = (double)perfTimer[ii].total - perfTimer[ii].average;
      sendBuf[ii] = temp * temp;
   }
   addDoubleParallel(sendBuf, recvBuf, numberOfTimers);
   for (int ii = 0; ii < numberOfTimers; ii++)
   {
      perfTimer[ii].stdev = sqrt(recvBuf[ii] / (double) getNRanks());
   }
}
