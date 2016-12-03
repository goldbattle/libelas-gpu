#include "elas_gpu.h"

using namespace std;



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
  //int32_t* d_P;
  //float* d_D;
  //uint8_t* d_I1, d_I2;
  //cudaMalloc((void**) &d_P, size*sizeof(float));
  //cudaMemcpy(d_A, h_A, size*sizeof(float), cudaMemcpyHostToDevice);

  
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



    // Calculate size of kernel
    int NT = 64;
    int N = ceil(to_calc.size()/NT) * NT;
    dim3 dimBlock(NT,1,1);
    dim3 dimGrid(max((int)(N/NT),1),1,1);
    // cout << "Cuda Elem Size: " << to_calc.size() << endl;
    // cout << "Cuda Block Size: " << NT << endl;
    // cout << "Cuda Grid Size: " << max((int)(N/NT),1) << endl;


    // Next launch our CUDA kernel
    // TODO: Convert this to CUDA kernel
    for(size_t j=0; j < to_calc.size(); j++) {
      int u = to_calc.at(j).first;
      int v = to_calc.at(j).second;
      // CPU Method
      findMatch(u,v,plane_a,plane_b,plane_c,disparity_grid,grid_dims,I1_desc,I2_desc,P,plane_radius,valid,right_image,D);
      // GPU Method
      // findMatch<<<dimGrid, dimBlock>>>(u,v,plane_a,plane_b,plane_c,disparity_grid,grid_dims,
      //                                   d_I1,d_I2,d_P,plane_radius,valid,right_image,d_D);
      // cudaDeviceSynchronize();
    }

    
  }

  // Copy the final disparity values back over
  // cudaMemcpy(h_C, d_C, size*sizeof(float), cudaMemcpyDeviceToHost);


  // Free local memory
  delete[] P;


  // Free cuda memory
  // cudaFree(d_A);

}



/**
 * CUDA Kernel for computing the match for a single UV coordinate
 */
__device__ void findMatch (int32_t &u,int32_t &v,float &plane_a,float &plane_b,float &plane_c,
                         int32_t* disparity_grid,int32_t *grid_dims,uint8_t* I1_desc,uint8_t* I2_desc,
                         int32_t *P,int32_t &plane_radius, bool valid, bool right_image, float* D) {









}