#ifndef __ELAS_GPU_H__
#define __ELAS_GPU_H__

// Enable profiling
#define PROFILE

#include <algorithm>
#include <math.h>
#include <vector>
#include <cuda.h>
#include <stdint.h>

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
public:

  // matching
  // __device__ void updatePosteriorMinimum (__m128i* I2_block_addr,const int32_t &d,const int32_t &w,
  //                                     const __m128i &xmm1,__m128i &xmm2,int32_t &val,int32_t &min_val,int32_t &min_d);
  // __device__ void updatePosteriorMinimum (__m128i* I2_block_addr,const int32_t &d,
  //                                     const __m128i &xmm1,__m128i &xmm2,int32_t &val,int32_t &min_val,int32_t &min_d);
  // __global__ void findMatch_GPU (int32_t* u_vals, int32_t* v_vals, float plane_a,float plane_b,float plane_c,
  //                        int32_t* disparity_grid,int32_t *grid_dims,uint8_t* I1_desc,uint8_t* I2_desc,
  //                        int32_t *P,int32_t plane_radius, int32_t width, int32_t height, bool valid, bool right_image, float* D);
  void computeDisparity(std::vector<support_pt> p_support,std::vector<triangle> tri,int32_t* disparity_grid,int32_t *grid_dims,
                        uint8_t* I1_desc,uint8_t* I2_desc,bool right_image,float* D);
  
  // __device__ uint32_t getAddressOffsetImage_GPU (const int32_t& u,const int32_t& v,const int32_t& width);

  // __device__ uint32_t getAddressOffsetGrid_GPU (const int32_t& x,const int32_t& y,const int32_t& d,const int32_t& width,const int32_t& disp_num);
};


#endif //__ELAS_GPU_H__
