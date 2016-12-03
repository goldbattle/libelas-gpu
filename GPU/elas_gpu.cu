#include "elas_gpu.h"

using namespace std;


void ElasGPU::removeInconsistentSupportPoints (int16_t* D_can,int32_t D_can_width,int32_t D_can_height) {
  
  cout << "Lower removeInconsistentSupportPoints has been called~!!!" << endl;


  // for all valid support points do
  for (int32_t u_can=0; u_can<D_can_width; u_can++) {
    for (int32_t v_can=0; v_can<D_can_height; v_can++) {
      int16_t d_can = *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width));
      //If the point has a calulated disparity
      if (d_can>=0) {
        // compute number of other points supporting the current point
        int32_t support = 0;
        //Checks a 5 pixel window for inconsistent disparities
        for (int32_t u_can_2=u_can-param.incon_window_size; u_can_2<=u_can+param.incon_window_size; u_can_2++) {
          for (int32_t v_can_2=v_can-param.incon_window_size; v_can_2<=v_can+param.incon_window_size; v_can_2++) {
            //Check we're inside candidate array (slightly smaller than image)
            if (u_can_2>=0 && v_can_2>=0 && u_can_2<D_can_width && v_can_2<D_can_height) {
              int16_t d_can_2 = *(D_can+getAddressOffsetImage(u_can_2,v_can_2,D_can_width));
              //Check if the disparity is different above a given threshold (5 pixels)
              //If it is considered fine, similar to other pixels around it, consider it a support
              if (d_can_2>=0 && abs(d_can-d_can_2)<=param.incon_threshold)
                support++;
            }
          }
        }
        
        // invalidate support point if number of supporting points is too low
        if (support<param.incon_min_support)
          *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width)) = -1;
      }
    }
  }
}