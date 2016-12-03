#ifndef __ELAS_GPU_H__
#define __ELAS_GPU_H__

// Enable profiling
#define PROFILE

#include <algorithm>
#include <math.h>
#include "elas.h"
#include "descriptor.h"
#include "triangle.h"
#include "matrix.h"


/**
 * Our ElasGPU class with all cuda implementations
 * Note where we extend the Elas class so we are calling
 * On all non-gpu functions there if they are not implemented
 */
class ElasGPU : public Elas {

public:

  // Constructor, input: parameters
  // Pass this to the super constructor
  ElasGPU(parameters param) : Elas(param) {}

// This was originally "private"
// Was converted to allow sub-classes to call this
// This assumes the user knows what they are doing
protected:

  // Override super
  void removeInconsistentSupportPoints (int16_t* D_can,int32_t D_can_width,int32_t D_can_height);

};


#endif //__ELAS_GPU_H__
