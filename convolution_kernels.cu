#include "blob.h"
#include "convolution.h"
#include "logging.h"
#include "timer.h"
#include <math.h>


void convolve_cpu(BLOB* in,BLOB* out,BLOB* w,int Kx,int Ky, conv_param_t* conv_param)
{
    for(int group_id=0;group_id<conv_param->group;group_id++)
    {
    int delta = (out->d/conv_param->group);//Depth of output divided by number of groups. 
    int output_starting_depth = group_id*delta;
    for(int out_depth=output_starting_depth;out_depth< output_starting_depth + delta;out_depth++)
        {
        int delta = (in->d/conv_param->group);//Depth of input divided by number of groups. 
        int in_depth_start = group_id*delta;
        for(int in_depth=in_depth_start;in_depth<in_depth_start+delta;in_depth++)
            {
            for(int out_y=0;out_y<out->h;out_y++)
                for(int out_x=0;out_x<out->w;out_x++)
                    for(int ky=0;ky<Ky;ky++)
                        for(int kx=0;kx<Kx;kx++)
                        {
                            int in_y = out_y*conv_param->Sy+ky;
                            int in_x = out_x*conv_param->Sx+kx;

                            int weigth_y = in_depth-(group_id*(in->d/conv_param->group));
                            int weigth_x = ky*Kx + kx;

                            float input = blob_data(in, in_depth, in_y,in_x);
                            float weight = blob_data(w, out_depth, weigth_y, weigth_x);
                                
                            blob_data(out,out_depth,out_y,out_x)+= input*weight; 
                        }
            }
        }
    }          


}


__device__ int calc_blob_id(int z,int y,int x,int height,int width)
{
    return z * height * width + y * width + x;

}

        // More complex convolution, runs only once so not really worth optimizing 
__global__ void gpu_device_convolve_depth_parrallel
    (float* data_in,float * data_weight, float* data_out // Data
    ,int Sx,int Sy // Sizes ...
    ,int in_w,int in_h,int in_d // input blob dimensions
    ,int w_w,int w_h // weigth height and depth
    ,int out_w,int out_h,int out_d // output width and height
    ,int Ky,int Kx 
    ,int group
    ,int in_depth_max)
    {
    unsigned int out_x = blockIdx.z*blockDim.z+ threadIdx.z;  
    unsigned int out_y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int out_depth = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(out_depth < out_d)
    {
        int out_id = calc_blob_id(out_depth,out_y,out_x,out_h,out_w);            
        for(int in_depth=0;in_depth<in_depth_max;in_depth++)
        {            
            for(int ky=0;ky<Ky;ky++)
            {
                for(int kx=0;kx<Kx;kx++)
                {
                    int in_y = out_y*Sy+ky;
                    int in_x = out_x*Sx+kx;

                    int weigth_y = in_depth;
                    int weight_x = ky*Kx + kx;
                    
                    int weight_id = calc_blob_id(out_depth,weigth_y,weight_x,w_h,w_w);
                    int in_id = calc_blob_id(in_depth,in_y,in_x,in_h,in_w);
    
                    data_out[out_id] += data_weight[weight_id] * data_in[in_id]; 
                }
            }        
        }
    }
}
// Old 
// __global__ void gpu_device_convolve_depth_parrallel_simple
//     (float* data_in,float * data_weight, float* data_out // Data
//     ,int in_w,int in_h,int in_d // input blob dimensions
//     ,int w_w,int w_h // weigth height and depth
//     ,int out_w,int out_h,int out_d // output width and height
    
//     ,int in_depth_max)
//     {
//     unsigned int out_x = blockIdx.z*blockDim.z+ threadIdx.z;  
//     unsigned int out_y = blockIdx.y*blockDim.y + threadIdx.y;
//     unsigned int out_depth = blockIdx.x*blockDim.x + threadIdx.x;
    
//     if(out_depth < out_d)
//     {
//         int out_id = calc_blob_id(out_depth,out_y,out_x,out_h,out_w);            
//         for(int in_depth=0;in_depth<in_depth_max;in_depth++)
//         {            
//             int weight_id = calc_blob_id(out_depth,in_depth,0,w_h,w_w);
//             int in_id = calc_blob_id(in_depth,out_y,out_x,in_h,in_w);

//             data_out[out_id] += data_weight[weight_id] * data_in[in_id]; 
//         }
//     }
// }

/*
// multiplexing width and height may reduce the cost of address calculation
// This kernel is still the most expensive, and runs often
// input width and height is always equal to output width and height
__global__ void gpu_device_convolve_depth_parrallel_simple_height_width_multiplexed
    (float* data_in,float * data_weight, float* data_out // Data
    ,int w_h // weigth height and depth
    ,int in_out_wh,int out_d // input/output width * height, output depth    
    ,int in_depth_max)
    {
    unsigned int in_out_xy = blockIdx.y*blockDim.y+ threadIdx.y;// 2d -> 1d pixel adress  
    unsigned int out_depth = blockIdx.x*blockDim.x + threadIdx.x;
	
    if(out_depth < out_d && in_out_xy < in_out_wh)
    {
		float local_out;
        int out_id = out_depth * in_out_wh + in_out_xy;            
        int in_id = in_out_xy; // per depth the same input values are loaded
        int weight_id = out_depth * w_h; // Weigth is different per depth
            
		local_out = data_out[out_id];
        for(int in_depth=0;in_depth<in_depth_max;in_depth++)
        {            
            local_out += data_weight[weight_id] * data_in[in_id]; 
            in_id+=in_out_wh;
            weight_id++;
        }
		data_out[out_id] = local_out;
    }
}
*/

// multiplexing width and height may reduce the cost of address calculation
// This kernel is still the most expensive, and runs often
// input width and height is always equal to output width and height
__global__ void gpu_device_convolve_depth_parrallel_simple_height_width_multiplexed_shared_mem
    (float* data_in,float * data_weight, float* data_out // Data
    ,int w_h // weigth height and depth
    ,int in_out_wh,int out_d // input/output width * height, output depth    
    ,int in_depth_max)
    {
    unsigned int in_out_xy = blockIdx.y*blockDim.y+ threadIdx.y;// 2d -> 1d pixel adress  
    unsigned int out_depth = blockIdx.x*blockDim.x + threadIdx.x;
    
	int max_shared_mem_size = 0;//(blockIdx.y*blockDim.y+ threadIdx.y);//49152/sizeof(float);


	
    if(out_depth < out_d && in_out_xy < in_out_wh)
    {

		__shared__ float shared_out[12288];//shared memory size = 12288
		
		float local_out; //local storage of data output
        int out_id = out_depth * in_out_wh + in_out_xy;            
        int in_id = in_out_xy; // per depth the same input values are loaded
        int weight_id = out_depth * w_h; // Weigth is different per depth
          
		if (out_id < in_out_wh && out_id < 12288)
			shared_out[out_id] = data_out[out_id];
		
		__syncthreads();
		
		if (out_id >= 12288){
			local_out = data_out[out_id];
		}
		else{
			local_out = shared_out[out_id];
		}
		
        for(int in_depth=0;in_depth<in_depth_max;in_depth++)
        {   
			local_out += data_weight[weight_id] * data_in[in_id]; 
			in_id+=in_out_wh;
            weight_id++; 
        }
		data_out[out_id] = local_out;
    }
}



// Runs a lot but already is relative quick
__global__ void gpu_device_convolve_naive_group_parrallel
    (float* data_in,float * data_weight, float* data_out // Data
    ,int Sx,int Sy // Sizes ...
    ,int in_w,int in_h,int in_d // input blob dimensions
    ,int w_w,int w_h // weigth height and depth
    ,int out_w,int out_h,int out_d // output width and height
    ,int Ky,int Kx 
    ,int group)
    {
    unsigned int out_x = blockIdx.z*blockDim.z+ threadIdx.z;  
    unsigned int out_y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int group_id = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(group_id < group)
    {
        int out_id = calc_blob_id(group_id,out_y,out_x,out_h,out_w);            
        
        for(int ky=0;ky<Ky;ky++)
        {
            for(int kx=0;kx<Kx;kx++)
            {
                int in_y = out_y*Sy+ky;
                int in_x = out_x*Sx+kx;

                int weigth_y = group_id-(group_id*(in_d/group));
                int weight_x = ky*Kx + kx;
                
                int weight_id = calc_blob_id(group_id,weigth_y,weight_x,w_h,w_w);
                int in_id = calc_blob_id(group_id,in_y,in_x,in_h,in_w);

                data_out[out_id] += data_weight[weight_id] * data_in[in_id]; 
            }
        }        
    }
}


int get_next_pow2(int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;

}

int ceil_div(int val,int div)
{
    int result = val/div;
    if(result * div < val) ++result;
    return result;
}

int _max(int val1,int val2)
{
    if(val1 > val2)
    {
        return val1;
    }
    else
    {
        return val2;
    }
}
int _min(int val1,int val2)
{
    if(val1 < val2)
    {
        return val1;
    }
    else
    {
        return val2;
    }
}
// HERE IT STARTS
void convolve_gpu(BLOB* in,BLOB* out,BLOB* w,int Kx,int Ky, conv_param_t* conv_param)
{
  timer_start();
  int in_depth_max = in->d/conv_param->group;//Depth of input divided by number of groups. 
  int out_depth_max = out->d/conv_param->group;//Depth of output divided by number of groups. 

  float* in_data;
  float* out_data;    
  float* w_data;
  
  blob2gpu(in_data, in);
  blob2gpu(out_data, out);
  blob2gpu(w_data, w); 


  int threadsPerBlockX = 1;
  int numBlocksX= 1;
  int threadsPerBlockYZ = 1;
  int numBlocksYZ = 1;
  // Can we ignore the group for loop?
  if(conv_param->group == 1)
  {

    
    // Can we ignore all these loop?
    if(Ky == 1 && Kx == 1 && conv_param->Sx == 1 && conv_param->Sy == 1)
    {

        threadsPerBlockX = _min(out_depth_max,1024);
        numBlocksX= ceil_div(out_depth_max,threadsPerBlockX);
        numBlocksYZ = ceil_div(out->h,threadsPerBlockYZ);
          
        threadsPerBlockYZ = 1024/threadsPerBlockX;
        numBlocksYZ = ceil_div(out->w * out->w, threadsPerBlockYZ);

        if(out->d == 1)
        {
            threadsPerBlockYZ =  _min(1024,out->w * out->w);
            numBlocksYZ = ceil_div(out->w * out->w,threadsPerBlockYZ);

            threadsPerBlockX = 1;
            numBlocksX = 1;
        }
        if(out->w == 1)
        {
            numBlocksYZ = 1;
            threadsPerBlockYZ=1;
        }

        dim3 grid( numBlocksX,numBlocksYZ, 1 );
        dim3 block(threadsPerBlockX, threadsPerBlockYZ, 1); 

		
		// Simplest yet slowest convolution
		gpu_device_convolve_depth_parrallel_simple_height_width_multiplexed_shared_mem<<<grid,block>>>(
		in_data,w_data,out_data
		,w->h
		,out->w*out->h,out->d
		,in_depth_max);	

           
    }  
    else
    {
        //return convolve_cpu(in,out,w,Kx,Ky,conv_param);
  

        numBlocksX=16;
        numBlocksYZ = 7;
        threadsPerBlockX = get_next_pow2(out_depth_max/numBlocksX+1);
        threadsPerBlockYZ =out->h/numBlocksYZ;
        
        dim3 grid( numBlocksX,numBlocksYZ, numBlocksYZ );
        dim3 block(threadsPerBlockX, threadsPerBlockYZ, threadsPerBlockYZ); 
        
        gpu_device_convolve_depth_parrallel<<<grid,block>>>(
            in_data,w_data,out_data
            ,conv_param->Sx,conv_param->Sy
            ,in->w,in->h,in->d
            ,w->w,w->h
            ,out->w,out->h,out->d
            ,Ky,Kx      
            ,conv_param->group
            ,in_depth_max);
    }


        
  }
  else
  {
    numBlocksX=16;
    numBlocksYZ = 7;
    threadsPerBlockYZ =out->h/numBlocksYZ;
    threadsPerBlockX = get_next_pow2(conv_param->group/numBlocksX+1);

    if(out->w == 1)
    {
        numBlocksYZ = 1;
        threadsPerBlockYZ=1;
    }
    dim3 grid( numBlocksX,numBlocksYZ, numBlocksYZ );          
    dim3 block(threadsPerBlockX, threadsPerBlockYZ, threadsPerBlockYZ); 
  
    //
    gpu_device_convolve_naive_group_parrallel<<<grid,block>>>(
              in_data,w_data,out_data
              ,conv_param->Sx,conv_param->Sy
              ,in->w,in->h,in->d
              ,w->w,w->h
              ,out->w,out->h,out->d
              ,Ky,Kx
              ,conv_param->group
              );

//   }
    
}

#ifdef DEBUG
printf("groups : %i \n",conv_param->group);
printf("out_width %i, out_height %i , out_depth_max : %i \n",out->w,out->h,out_depth_max);
printf("in_width %i, in_height %i , in_depth_max : %i \n",in->w,in->h,in_depth_max);
printf("Kx : %i, Ky : %i , Sx : %i ,Sy : %i \n",Kx,Ky,conv_param->Sx,conv_param->Sy);
int threads_per_block = threadsPerBlockX * threadsPerBlockYZ * threadsPerBlockYZ;
if(conv_param->group == 1 && Ky == 1 && Kx == 1 && conv_param->Sx == 1 && conv_param->Sy == 1)
  {
    //
        printf("GRID : (x : %i) (y : % i) (z : %i) , ",numBlocksX,numBlocksYZ,1);

        threads_per_block = threadsPerBlockX * threadsPerBlockYZ;
        printf("BLOCK : (x : %i) (y : % i) (z : %i), (total tpb : %i) \n",threadsPerBlockX,threadsPerBlockYZ,1,threads_per_block);

    
}
else
{
  printf("BLOCK : (x : %i) (y : % i) (z : %i), (total tpb : %i) \n",threadsPerBlockX,threadsPerBlockYZ,threadsPerBlockYZ,threads_per_block);
  printf("GRID : (x : %i) (y : % i) (z : %i) , ",numBlocksX,numBlocksYZ,numBlocksYZ);

}
if(threads_per_block > 1024)
{
    printf("TOO MANY THREADS PER BLOCK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");    
}

#endif
 
  gpu2blob(out,out_data);

  cudaFree(in_data);
  cudaFree(out_data);
  cudaFree(w_data);
  if(conv_param->group == 1)
  {
    if(Ky == 1 && Kx == 1 && conv_param->Sx == 1 && conv_param->Sy == 1)
    {
        writeToFile("depth_Parrallel_simple",(double)timer_stop());
    }
    else
    {
        writeToFile("depth_Parrallel_complex",(double)timer_stop());
    }
  }
  else
  {
    writeToFile("group_Parrallel",(double)timer_stop());
      
  }
  timer_destroy();

}   