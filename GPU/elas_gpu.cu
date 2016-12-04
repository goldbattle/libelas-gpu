#include "elas_gpu.h"

using namespace std;

__device__ uint32_t getAddressOffsetImage_GPU (const int32_t& u,const int32_t& v,const int32_t& width) {
  return v*width+u;
}

__device__ uint32_t getAddressOffsetGrid_GPU (const int32_t& x,const int32_t& y,const int32_t& d,const int32_t& width,const int32_t& disp_num) {
  return (y*width+x)*disp_num+d;
}

/**
 * CUDA Kernel for computing the match for a single UV coordinate
 */
__global__ void findMatch_GPU (int32_t* u_vals, int32_t* v_vals, int32_t size_vals, float plane_a, float plane_b, float plane_c,
                         int32_t* disparity_grid,int32_t *grid_dims,uint8_t* I1_desc,uint8_t* I2_desc,
                         int32_t* P, int32_t plane_radius, int32_t width ,int32_t height, bool valid, bool right_image, float* D) {
 
  // get image width and height
  const int32_t disp_num    = grid_dims[0]-1;
  const int32_t window_size = 2;

  
  //TODO: Remove hard code and use param
  bool subsampling = false;
  bool match_texture = false;
  int32_t grid_size = 20;

  // Pixel id
  uint32_t idx = blockDim.x*blockIdx.x + threadIdx.x;

  // Check that we are in range
  if(idx >= size_vals)
    return;

  // Else get our value
  uint32_t u = u_vals[idx];
  uint32_t v = v_vals[idx];

  // address of disparity we want to compute
  uint32_t d_addr;
  if (subsampling) d_addr = getAddressOffsetImage_GPU(u/2,v/2,width/2);
  else             d_addr = getAddressOffsetImage_GPU(u,v,width);
  
  // check if u is ok
  if (u<window_size || u>=width-window_size)
    return;

  // compute line start address
  int32_t  line_offset = 16*width*max(min(v,height-3),2);
  uint8_t *I1_line_addr,*I2_line_addr;
  if (!right_image) {
    I1_line_addr = I1_desc+line_offset;
    I2_line_addr = I2_desc+line_offset;
  } else {
    I1_line_addr = I2_desc+line_offset;
    I2_line_addr = I1_desc+line_offset;
  }

  // compute I1 block start address
  uint8_t* I1_block_addr = I1_line_addr+16*u;
  
  // does this patch have enough texture?
  int32_t sum = 0;
  for (int32_t i=0; i<16; i++)
    sum += abs((int32_t)(*(I1_block_addr+i))-128);
  if (sum<match_texture)
    return;

  // compute disparity, min disparity and max disparity of plane prior
  int32_t d_plane     = (int32_t)(plane_a*(float)u+plane_b*(float)v+plane_c);
  int32_t d_plane_min = max(d_plane-plane_radius,0);
  int32_t d_plane_max = min(d_plane+plane_radius,disp_num-1);

  // get grid pointer
  int32_t  grid_x    = (int32_t)floor((float)u/(float)grid_size);
  int32_t  grid_y    = (int32_t)floor((float)v/(float)grid_size);
  uint32_t grid_addr = getAddressOffsetGrid_GPU(grid_x,grid_y,0,grid_dims[1],grid_dims[0]);  
  int32_t  num_grid  = *(disparity_grid+grid_addr);
  int32_t* d_grid    = disparity_grid+grid_addr+1;
  
  // loop variables
  int32_t d_curr, u_warp, val;
  int32_t min_val = 10000;
  int32_t min_d   = -1;
  //__m128i xmm1    = _mm_load_si128((__m128i*)I1_block_addr);
  //__m128i xmm2;

  // left image
  if (!right_image) { 
    for (int32_t i=0; i<num_grid; i++) {
      d_curr = d_grid[i];
      if (d_curr<d_plane_min || d_curr>d_plane_max) { //If the current disparity is out of the planes range
        u_warp = u-d_curr;
        if (u_warp<window_size || u_warp>=width-window_size)
          continue;
        //updatePosteriorMinimum((__m128i*)(I2_line_addr+16*u_warp),d_curr,xmm1,xmm2,val,min_val,min_d);
        val = 0;
        for(int j=0; j<16; j++){
            val += abs((uint32_t*)(I1_block_addr+j)-(uint32_t*)(I2_line_addr+j+16*u_warp));
        }
        // xmm2 = _mm_load_si128((__m128i*)(I2_line_addr+16*u_warp));
        // xmm2 = _mm_sad_epu8(xmm1,xmm2);
        // val  = _mm_extract_epi16(xmm2,0)+_mm_extract_epi16(xmm2,4);
        if (val<min_val) {
            min_val = val;
            min_d   = d_curr;
        }
      }
    }
    //disparity inside the grid
    for (d_curr=d_plane_min; d_curr<=d_plane_max; d_curr++) {
      u_warp = u-d_curr;
      if (u_warp<window_size || u_warp>=width-window_size)
        continue;
      //   updatePosteriorMinimum((__m128i*)(I2_line_addr+16*u_warp),d_curr,valid?*(P+abs(d_curr-d_plane)):0,xmm1,xmm2,val,min_val,min_d);
      val = 0;
      for(int j=0; j<16; j++){
          val += abs((uint32_t*)(I1_block_addr+j)-(uint32_t*)(I2_line_addr+j+16*u_warp) + valid?*(P+abs(d_curr-d_plane)):0);
      }
      //   xmm2 = _mm_load_si128(I2_block_addr);
      //   xmm2 = _mm_sad_epu8(xmm1,xmm2);
      //   val  = _mm_extract_epi16(xmm2,0)+_mm_extract_epi16(xmm2,4)+w;
      if (val<min_val) {
        min_val = val;
        min_d   = d_curr;
      }
    }
    
  // right image
  } else {
    for (int32_t i=0; i<num_grid; i++) {
      d_curr = d_grid[i];
      if (d_curr<d_plane_min || d_curr>d_plane_max) {
        u_warp = u+d_curr;
        if (u_warp<window_size || u_warp>=width-window_size)
          continue;
        //updatePosteriorMinimum((__m128i*)(I2_line_addr+16*u_warp),d_curr,xmm1,xmm2,val,min_val,min_d);
        val = 0;
        for(int j=0; j<16; j++){
            val += abs((uint32_t*)(I1_block_addr+j)-(uint32_t*)(I2_line_addr+j+16*u_warp));
        }
        // xmm2 = _mm_load_si128((__m128i*)(I2_line_addr+16*u_warp));
        // xmm2 = _mm_sad_epu8(xmm1,xmm2);
        // val  = _mm_extract_epi16(xmm2,0)+_mm_extract_epi16(xmm2,4);
        if (val<min_val) {
            min_val = val;
            min_d   = d_curr;
        }
      }
    }
    for (d_curr=d_plane_min; d_curr<=d_plane_max; d_curr++) {
      u_warp = u+d_curr;
      if (u_warp<window_size || u_warp>=width-window_size)
        continue;
      //   updatePosteriorMinimum((__m128i*)(I2_line_addr+16*u_warp),d_curr,valid?*(P+abs(d_curr-d_plane)):0,xmm1,xmm2,val,min_val,min_d);
      val = 0;
      for(int j=0; j<16; j++){
          val += abs((uint32_t*)(I1_block_addr+j)-(uint32_t*)(I2_line_addr+j+16*u_warp) + valid?*(P+abs(d_curr-d_plane)):0);
      }
      //   xmm2 = _mm_load_si128(I2_block_addr);
      //   xmm2 = _mm_sad_epu8(xmm1,xmm2);
      //   val  = _mm_extract_epi16(xmm2,0)+_mm_extract_epi16(xmm2,4)+w;
      if (val<min_val) {
        min_val = val;
        min_d   = d_curr;
      }
    }
  }

  // set disparity value
  if (min_d>=0) *(D+d_addr) = min_d; // MAP value (min neg-Log probability)
  else          *(D+d_addr) = -1;    // invalid disparity
}

/**
 * This is the core method that computes the disparity of the image
 * It processes each triangle, so we create a kernel and have each thread
 * compute the matches in each triangle
 */
void ElasGPU::computeDisparity(std::vector<support_pt> p_support,std::vector<triangle> tri,int32_t* disparity_grid,int32_t *grid_dims,
                                uint8_t* I1_desc,uint8_t* I2_desc,bool right_image,float* D) {

  // number of disparities
  const int32_t disp_num  = grid_dims[0]-1;
  
  // descriptor window_size
  int32_t window_size = 2;
  
  // init disparity image to -10
  if (param.subsampling) {
    for (int32_t i=0; i<(width/2)*(height/2); i++)
      *(D+i) = -10;
  } else {
    for (int32_t i=0; i<width*height; i++)
      *(D+i) = -10;
  }
  
  // pre-compute prior 
  float two_sigma_squared = 2*param.sigma*param.sigma;
  int32_t* P = new int32_t[disp_num];
  for (int32_t delta_d=0; delta_d<disp_num; delta_d++)
    P[delta_d] = (int32_t)((-log(param.gamma+exp(-delta_d*delta_d/two_sigma_squared))+log(param.gamma))/param.beta);
  int32_t plane_radius = (int32_t)max((float)ceil(param.sigma*param.sradius),(float)2.0);

  // loop variables
  int32_t c1, c2, c3;
  float plane_a,plane_b,plane_c,plane_d;


  // CUDA copy over needed memory information
  // disparity_grid, I1_desc,I2_desc,P,D
  int32_t* d_disparity_grid, *d_grid_dims;
  int32_t* d_P;
  float* d_D;
  uint8_t* d_I1, *d_I2;
  //Allocate on global memory
  cudaMalloc((void**) &d_disparity_grid, grid_dims[0]*grid_dims[1]*grid_dims[2]*sizeof(int32_t));
  cudaMalloc((void**) &d_P, disp_num*sizeof(int32_t));
  cudaMalloc((void**) &d_D, width*height*sizeof(float));
  cudaMalloc((void**) &d_I1, 16*width*height*sizeof(uint8_t)); //Device descriptors
  cudaMalloc((void**) &d_I2, 16*width*height*sizeof(uint8_t)); //Device descriptors
  cudaMalloc((void**) &d_grid_dims, 3*sizeof(int32_t)); 

  //Now copy over data
  cudaMemcpy(d_disparity_grid, disparity_grid, grid_dims[0]*grid_dims[1]*grid_dims[2]*sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_P, P, disp_num*sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_D, D, width*height*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_I1, I1_desc, 16*width*height*sizeof(uint8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_I2, I2_desc, 16*width*height*sizeof(uint8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_grid_dims, grid_dims, 3*sizeof(int32_t), cudaMemcpyHostToDevice); 

  // for all triangles do
  for (uint32_t i=0; i<tri.size(); i++) {
    
    // get plane parameters
    uint32_t p_i = i*3;
    if (!right_image) {
      plane_a = tri[i].t1a;
      plane_b = tri[i].t1b;
      plane_c = tri[i].t1c;
      plane_d = tri[i].t2a;
    } else {
      plane_a = tri[i].t2a;
      plane_b = tri[i].t2b;
      plane_c = tri[i].t2c;
      plane_d = tri[i].t1a;
    }
    
    // triangle corners
    c1 = tri[i].c1;
    c2 = tri[i].c2;
    c3 = tri[i].c3;

    // sort triangle corners wrt. u (ascending)    
    float tri_u[3];
    if (!right_image) {
      tri_u[0] = p_support[c1].u;
      tri_u[1] = p_support[c2].u;
      tri_u[2] = p_support[c3].u;
    } else {
      tri_u[0] = p_support[c1].u-p_support[c1].d;
      tri_u[1] = p_support[c2].u-p_support[c2].d;
      tri_u[2] = p_support[c3].u-p_support[c3].d;
    }
    float tri_v[3] = {p_support[c1].v,p_support[c2].v,p_support[c3].v};
    
    for (uint32_t j=0; j<3; j++) {
      for (uint32_t k=0; k<j; k++) {
        if (tri_u[k]>tri_u[j]) {
          float tri_u_temp = tri_u[j]; tri_u[j] = tri_u[k]; tri_u[k] = tri_u_temp;
          float tri_v_temp = tri_v[j]; tri_v[j] = tri_v[k]; tri_v[k] = tri_v_temp;
        }
      }
    }
    
    // rename corners
    float A_u = tri_u[0]; float A_v = tri_v[0];
    float B_u = tri_u[1]; float B_v = tri_v[1];
    float C_u = tri_u[2]; float C_v = tri_v[2];
    
    // compute straight lines connecting triangle corners
    float AB_a = 0; float AC_a = 0; float BC_a = 0;
    if ((int32_t)(A_u)!=(int32_t)(B_u)) AB_a = (A_v-B_v)/(A_u-B_u);
    if ((int32_t)(A_u)!=(int32_t)(C_u)) AC_a = (A_v-C_v)/(A_u-C_u);
    if ((int32_t)(B_u)!=(int32_t)(C_u)) BC_a = (B_v-C_v)/(B_u-C_u);
    float AB_b = A_v-AB_a*A_u;
    float AC_b = A_v-AC_a*A_u;
    float BC_b = B_v-BC_a*B_u;
    
    // a plane is only valid if itself and its projection
    // into the other image is not too much slanted
    bool valid = fabs(plane_a)<0.7 && fabs(plane_d)<0.7;

    // Vector of all u,v pairs we need to calculate
    std::vector<std::pair<int32_t,int32_t>> to_calc;
        
    // first part (triangle corner A->B)
    if ((int32_t)(A_u)!=(int32_t)(B_u)) {
      // Starting at A_u loop till the B_u or the end of the image
      for (int32_t u=max((int32_t)A_u,0); u<min((int32_t)B_u,width); u++){
        // If we are sub-sampling skip every two
        if (!param.subsampling || u%2==0) {
          // Use linear lines, to get the bounds of where we need to check
          int32_t v_1 = (uint32_t)(AC_a*(float)u+AC_b);
          int32_t v_2 = (uint32_t)(AB_a*(float)u+AB_b);
          // Loop through these values of v and try to find the match
          for (int32_t v=min(v_1,v_2); v<max(v_1,v_2); v++)
            // If we are sub-sampling skip every two
            if (!param.subsampling || v%2==0) {
              to_calc.push_back(std::pair<int32_t,int32_t>(u,v));
            }
        }
      }
    }

    // second part (triangle corner B->C)
    if ((int32_t)(B_u)!=(int32_t)(C_u)) {
      for (int32_t u=max((int32_t)B_u,0); u<min((int32_t)C_u,width); u++){
        if (!param.subsampling || u%2==0) {
          int32_t v_1 = (uint32_t)(AC_a*(float)u+AC_b);
          int32_t v_2 = (uint32_t)(BC_a*(float)u+BC_b);
          for (int32_t v=min(v_1,v_2); v<max(v_1,v_2); v++)
            if (!param.subsampling || v%2==0) {
              to_calc.push_back(std::pair<int32_t,int32_t>(u,v));
            }
        }
      }
    }

    int size = to_calc.size()*sizeof(int32_t);
    // Convert vector to array
    int32_t* u_vals = (int32_t*)malloc(size);
    int32_t* v_vals = (int32_t*)malloc(size);

    // Save to arrays
    for(size_t j=0; j < to_calc.size(); j++) {
      u_vals[j] = to_calc.at(j).first;
      v_vals[j] = to_calc.at(j).second;
    }

    // Copy to device code
    int32_t* d_u_vals, *d_v_vals;
    cudaMalloc((void**) &d_u_vals, size);
    cudaMalloc((void**) &d_v_vals, size);
    cudaMemcpy(d_u_vals, u_vals, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_vals, v_vals, size, cudaMemcpyHostToDevice);

    // Calculate size of kernel
    int block_size = 32;
    int grid_size = 0;
    //Calculate gridsize (Add 1 if not evenly divided)
    if(to_calc.size()%block_size == 0){
        grid_size = ceil(to_calc.size()/block_size);
    }else{
        grid_size = ceil(to_calc.size()/block_size) + 1;
    }

    dim3 DimGrid(grid_size,1,1);
    dim3 DimBlock(block_size,1,1);

    // cout << "Cuda Elem Size: " << to_calc.size() << endl;
    // cout << "Cuda Block Size: " << block_size << endl;
    // cout << "Cuda Grid Size: " << grid_size << endl;

    // Next launch our CUDA kernel
    // TODO: Convert this to CUDA kernel
    // for(size_t j=0; j < to_calc.size(); j++) {
    //   int u = to_calc.at(j).first;
    //   int v = to_calc.at(j).second;
    //   // CPU Method
    //   findMatch(u,v,plane_a,plane_b,plane_c,disparity_grid,grid_dims,I1_desc,I2_desc,P,plane_radius,valid,right_image,D);
    // }

    //GPU Method
    findMatch_GPU<<<DimGrid, DimBlock>>>(d_u_vals,d_v_vals,to_calc.size(),plane_a,plane_b,plane_c,d_disparity_grid,d_grid_dims,
                                        d_I1,d_I2,d_P,plane_radius,width,height,valid,right_image,d_D);
    
    cudaDeviceSynchronize();
    cudaFree(d_u_vals);
    cudaFree(d_v_vals);
    
  }

  // Copy the final disparity values back over
  
  cudaMemcpy(disparity_grid, d_disparity_grid, grid_dims[0]*grid_dims[1]*grid_dims[2]*sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(D, d_D, width*height*sizeof(float), cudaMemcpyDeviceToHost);

  
  // Free local memory
  delete[] P;


  // Free cuda memory
  cudaFree(d_disparity_grid);
  cudaFree(d_P);
  cudaFree(d_D);
  cudaFree(d_I1);
  cudaFree(d_I2);
  cudaFree(d_grid_dims);

}