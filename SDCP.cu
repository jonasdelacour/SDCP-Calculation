#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <fstream>

/** Change this to double if you want double-precision, be wary this will significantly penalize performance **/
typedef float real_t;

/** Change model parameters here **/
constexpr real_t h_numMets = 2.0;
constexpr real_t h_fracClon = 0.5;
constexpr real_t h_minDetectable = 1e9;
size_t L = (size_t)(h_minDetectable * h_numMets * h_fracClon);
__constant__ real_t alpha = 0.136;
__constant__ real_t fracMets = 1.0;
__constant__ real_t SF2 = 0.5197; 
__constant__ real_t Nfx = 34.0;
__constant__ real_t PD = 57.0; 
__constant__ real_t sdcpPD = 0.9;

/** File Parameters **/
constexpr unsigned int max_arr_size = 160; //Change this if the maximum length of the data sets increases. 
constexpr size_t file_count = 168;
std::string path = "";
std::string csv_filenames[file_count] = {"46_TP2-90.csv","46_Baseline.csv","46_TP2-85.csv","46_NoCTV2.csv","91_TP2-90.csv","91_Baseline.csv","91_TP2-85.csv","91_NoCTV2.csv","BC_Baseline.csv","BC_NoCTV2.csv","BC_TP2-90.csv","BC_TP2-85.csv","Bo_NoCTV2.csv","Bo_TP2-90.csv","Bo_TP2-85.csv","Bo_Baseline.csv","E6_Baseline.csv","E6_TP2-90.csv","E6_TP2-85.csv","E6_NoCTV2.csv","Kb_TP2-85.csv","Kb_Baseline.csv","Kb_TP2-90.csv","Kb_NoCTV2.csv","LB_TP2-85.csv","LB_Baseline.csv","LB_TP2-90.csv","LB_NoCTV2.csv","N2_TP2-85.csv","N2_TP2-90.csv","N2_Baseline.csv","N2_NoCTV2.csv","QC_Baseline.csv","QC_TP2-90.csv","QC_TP2-85.csv","QC_NoCTV2.csv","Qa_TP2-90.csv","Qa_TP2-85.csv","Qa_Baseline.csv","Qa_NoCTV2.csv","RH_Baseline.csv","RH_TP2-90.csv","RH_TP2-85.csv","RH_NoCTV2.csv","RI_Baseline.csv","RI_TP2-85.csv","RI_TP2-90.csv","RI_NoCTV2.csv","V8_NoCTV2.csv","V8_TP2-90.csv","V8_TP2-85.csv","V8_Baseline.csv","WL_TP2-90.csv","WL_TP2-85.csv","WL_Baseline.csv","WL_NoCTV2.csv","Wj_Baseline.csv","Wj_TP2-85.csv","Wj_NoCTV2.csv","Wj_TP2-90.csv","ZV_TP2-90.csv","ZV_NoCTV2.csv","ZV_Baseline.csv","ZV_TP2-85.csv","Zl_TP2-90.csv","Zl_TP2-85.csv","Zl_NoCTV2.csv","Zl_Baseline.csv","aH_Baseline.csv","aH_NoCTV2.csv","aH_TP2-90.csv","aH_TP2-85.csv","aR_TP2-85.csv","aR_NoCTV2.csv","aR_Baseline.csv","aR_TP2-90.csv","aY_Baseline.csv","aY_TP2-85.csv","aY_NoCTV2.csv","aY_TP2-90.csv","b2_TP2-90.csv","b2_Baseline.csv","b2_TP2-85.csv","b2_NoCTV2.csv","b5_TP2-85.csv","b5_TP2-90.csv","b5_Baseline.csv","b5_NoCTV2.csv","bw_Baseline.csv","bw_TP2-90.csv","bw_TP2-85.csv","bw_NoCTV2.csv","de_Baseline.csv","de_TP2-90.csv","de_TP2-85.csv","de_NoCTV2.csv","eX_Baseline.csv","eX_TP2-90.csv","eX_TP2-85.csv","eX_NoCTV2.csv","f7_Baseline.csv","f7_NoCTV2.csv","f7_TP2-90.csv","f7_TP2-85.csv","fn_TP2-85.csv","fn_Baseline.csv","fn_NoCTV2.csv","fn_TP2-90.csv","h0_NoCTV2.csv","h0_TP2-85.csv","h0_TP2-90.csv","h0_Baseline.csv","hH_TP2-85.csv","hH_TP2-90.csv","hH_Baseline.csv","hH_NoCTV2.csv","hN_TP2-85.csv","hN_TP2-90.csv","hN_Baseline.csv","hN_NoCTV2.csv","hS_TP2-85.csv","hS_Baseline.csv","hS_NoCTV2.csv","hS_TP2-90.csv","hW_TP2-90.csv","hW_Baseline.csv","hW_TP2-85.csv","hW_NoCTV2.csv","i5_TP2-90.csv","i5_TP2-85.csv","i5_Baseline.csv","i5_NoCTV2.csv","iU_TP2-85.csv","iU_TP2-90.csv","iU_NoCTV2.csv","iU_Baseline.csv","jP_TP2-90.csv","jP_TP2-85.csv","jP_Baseline.csv","jP_NoCTV2.csv","kx_TP2-90.csv","kx_Baseline.csv","kx_NoCTV2.csv","kx_TP2-85.csv","mB_TP2-90.csv","mB_Baseline.csv","mB_NoCTV2.csv","mB_TP2-85.csv","nE_TP2-85.csv","nE_Baseline.csv","nE_NoCTV2.csv","nE_TP2-90.csv","re_TP2-85.csv","re_TP2-90.csv","re_NoCTV2.csv","re_Baseline.csv","tU_NoCTV2.csv","tU_TP2-85.csv","tU_Baseline.csv","tU_TP2-90.csv","yN_TP2-90.csv","yN_NoCTV2.csv","yN_Baseline.csv","yN_TP2-85.csv","zY_NoCTV2.csv","zY_TP2-90.csv","zY_TP2-85.csv","zY_Baseline.csv"};

/** Do not change these, the purpose of this is to allow both the CPU and the GPU to see these values, so they are first defined as constexpr for the CPU and then as __constant__ for the GPU. **/
__constant__ real_t numMets = h_numMets;
__constant__ real_t minDetectable = h_minDetectable; 
__constant__ real_t fracClon = h_fracClon;


namespace cg = cooperative_groups;
using namespace std::literals;

/** Function that allows me to evaluate the nearest power of 2 of a number at compile time thus avoiding a switch-case statement for launching the GPU-Kernel **/
constexpr
int minpow2(int v)
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

/** Device function for computing integer powers **/
__device__ real_t pow_int(real_t a, int exponent){
    real_t temp = a;
    #pragma unroll
    for (size_t i = 1; i < exponent; i++)
    {
        temp *= a;
    }
    return temp;
}

/** Single CUDA-block reduction method **/
template <unsigned int rounded_block_size>
__device__ real_t reduction(real_t *sdata, const size_t arr_length){

    /** Might need to mask off nans and infinites from the reduction function if the array is smaller than the block-size. **/
    /*
    if (threadIdx.x > arr_length)
    {
        sdata[threadIdx.x] = (real_t)0.0; 
    }*/
    __syncthreads();
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);
    sdata[threadIdx.x] = cg::reduce(tile32, sdata[threadIdx.x], cg::plus<real_t>());
    __syncthreads();

    if((rounded_block_size > 512)){if ((threadIdx.x < 512) && ((threadIdx.x % 32) == 0)){sdata[threadIdx.x] += sdata[threadIdx.x + 512];} __syncthreads();}
    if((rounded_block_size > 256)){if ((threadIdx.x < 256) && ((threadIdx.x % 32) == 0)){sdata[threadIdx.x] += sdata[threadIdx.x + 256];} __syncthreads();}
    if((rounded_block_size > 128)){if ((threadIdx.x < 128) && ((threadIdx.x % 32) == 0)){sdata[threadIdx.x] += sdata[threadIdx.x + 128];} __syncthreads();}
    if((rounded_block_size > 64)){if ((threadIdx.x < 64) && ((threadIdx.x % 32) == 0)){sdata[threadIdx.x] += sdata[threadIdx.x + 64];} __syncthreads();}
    if((rounded_block_size > 32)){if ((threadIdx.x < 32) && ((threadIdx.x % 32) == 0)){sdata[threadIdx.x] += sdata[threadIdx.x + 32];} __syncthreads();}

    __syncthreads();
    real_t sum = sdata[0];
    __syncthreads();
    return sum;

}

/** The kernel **/
template <unsigned int rounded_block_size>
__global__ void calcSDCP(   real_t* input_D, 
                            real_t* input_V,
                            real_t* results,
                            size_t* arr_size,
                            size_t L)
{   
    /** Pointer to Cache / L1 memory, the array size is specificed in the kernel launch **/
    extern __shared__ real_t smem[];

    /** Synchronization Handle **/ 
    cg::thread_block block_handle = cg::this_thread_block(); 

    /** Move D and V to thread-private registers for fast computation **/ 
    real_t D = input_D[threadIdx.x + blockIdx.x * blockDim.x];
    real_t V = input_V[threadIdx.x + blockIdx.x * blockDim.x];
    
    cg::sync(block_handle);
    
    /** Fill out cache with V values and use reduction method to sum **/
    smem[threadIdx.x] = V;
    real_t Vtot = reduction<rounded_block_size>(smem,arr_size[blockIdx.x]);
    real_t logL = __logf(L);
    real_t beta = (-log(SF2) - 2*alpha)/ (real_t)4.0;

    /** Kahan Sum Variables **/
    real_t c = (real_t)0.0;
    real_t sumL = (real_t)0.0;

    /** Core Computational Effort Starts Here **/
    
    for (size_t N = 1e8; N > 1; N--) /** The for loop is reversed because summing the smallest numbers first reduces total rounding error **/
    {
        smem[threadIdx.x] =  pow((1 - exp(-alpha* D - beta* (D* D)/Nfx)), N/numMets) * V/Vtot; //First assign the values to a shared cache that all threads in a thread-block can read, to prepare for reduction.
        real_t sum_term = reduction<rounded_block_size>(smem,arr_size[blockIdx.x]); //This is how you sum in parallel. Look up "Parallel Reduction" if curious.
        
        /** Kahan Summation Algorithm seems to recover as much or more than multi-layer partial sums, yet it is much faster **/
        real_t y = ( fracMets/(N* logL) * pow_int(sum_term , numMets) ) - c;
        real_t t = sumL + y;
        c = (t - sumL)- y;
        sumL = t;

        ///** Progress Printing **/
        //if ((N%1000000) == 0)
        //{
        //    if ((blockIdx.x * blockDim.x  + threadIdx.x )== 0)
        //    {
        //        printf("Progress: %d% \n", (int)( (((real_t)(L+1)-(real_t)N)/(real_t)L )* 100) );
        //    }
        //}
        
    }
    /** Writes sumL back into the VRAM global memory of the GPU such that it can be copied to RAM and read from the HOST(CPU)- Code **/
    if (threadIdx.x == 0)
    {
        results[blockIdx.x] = sumL + 1 - fracMets;
        if(blockIdx.x == 0){printf("Progress: 100% \n");}
    }
}

/** Reads 2 row CSV files **/
size_t read_csv(std::string filename, real_t array[]) {
    // Get filestream
    std::fstream filestream;
    filestream.open(filename, std::fstream::in);
    
    std::string line;
    int j = 0;
    
    // To read 2 lines, but probably fails if only 1 line..
    for (auto i = 0; i < 2; i++) {
    
    // To read all lines
    //while(std::getline(stream, line)) {
    
        std::getline(filestream, line);
        
        // Splitting and parsing
        std::string delimiter = ","; // Could be given as input or global var
        size_t last = 0;
        size_t next = 0;
        
        while ((next = line.find(delimiter, last)) != std::string::npos) {  
            array[j] = std::stof(line.substr(last, next-last));

            last = next + 1; 
            ++j;
        }
        array[j] = std::stof(line.substr(last));
        j++;
    }
    
    filestream.close();
    return (size_t)((j+1)/2);
}


int main(){
    /** Host Data arrays **/
    real_t* Data = new real_t[max_arr_size*2*file_count];
    real_t* D = new real_t[max_arr_size*file_count];
    real_t* V = new real_t[max_arr_size*file_count];
    real_t* results = new real_t[file_count];
    size_t* arr_sizes = new size_t[file_count];
    
    /** Device Pointers, GPU - memory addresses **/
    real_t* d_D;
    real_t* d_V;
    real_t* d_results;    
    size_t* d_arr_sizes;

    /** (no native filesystem::directory_iterator support in CUDA (C++17)) 
     * So use this array to specify file names, the results will be displayed in the same order as the file-array.**/
    
    for (size_t i = 0; i < file_count; i++)
    {
        csv_filenames[i] = path + csv_filenames[i];
    }
    
    /** Fill all csv data into 1 contiguous array **/
    for (size_t i = 0; i < file_count; ++i) {
        auto csv = csv_filenames[i];
        size_t length = read_csv(csv, &Data[max_arr_size*2*i]);
        arr_sizes[i] = length;
    }

    /** Splits data into D and V arrays **/
    for (size_t i = 0; i < file_count; i++)
    {
        for (size_t j = 0; j < max_arr_size; j++)
        {
            D[i*max_arr_size + j] = Data[i*2*max_arr_size + j];
            V[i*max_arr_size + j] = Data[i*2*max_arr_size + arr_sizes[i] + j];
        }   
    }

    /** Allocate Device Pointers **/
    cudaMalloc(&d_D, sizeof(real_t)*max_arr_size * file_count);
    cudaMalloc(&d_V, sizeof(real_t)*max_arr_size * file_count);
    cudaMalloc(&d_results, sizeof(real_t)*max_arr_size * file_count);
    cudaMalloc(&d_arr_sizes, sizeof(size_t)*file_count);
    
    /** Copy data to device **/
    cudaMemcpy(d_D, D, sizeof(real_t)*max_arr_size * file_count , cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, sizeof(real_t)*max_arr_size * file_count , cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_sizes, arr_sizes, sizeof(size_t)*file_count, cudaMemcpyHostToDevice);

    /** Launch Kernel **/
    auto start = std::chrono::system_clock::now();
    void* kernelArgs[] = {(void*)&d_D, (void*)&d_V, (void*)&d_results, (void*)&d_arr_sizes, (void*)&L};
    const void* fun = (void*)calcSDCP<max_arr_size*2>;
    /** The requirement of constant valued template arguments necessitates this monstrosity of a switch statement **/
    cudaLaunchCooperativeKernel((void*)calcSDCP<minpow2(max_arr_size)>, dim3(file_count,1,1), dim3(max_arr_size,1,1), kernelArgs, sizeof(real_t)*minpow2(max_arr_size)*4, NULL);
    cudaDeviceSynchronize();

    /** Copy the results back to the host **/
    cudaMemcpy((void*)results, d_results, sizeof(real_t)*file_count, cudaMemcpyDeviceToHost);

    /** Print the results **/
    printf("\nResults: [ %e",results[0]);
    for (size_t i = 1; i < file_count; i++)
    {
        printf(", %e", results[i]);
    }
    std::cout << "] \n";
    
    auto end = std::chrono::system_clock::now();
    std::cout << "Elapsed time: " << (end-start)/ 1ms << "ms\n" ;

    /** Prevents potential memory leak problems 
     * If you ever start experiencing seg-faults use the nvidia-smi command to see if there is some allocated memory that shouldn't be
     * Otherwise cuda-memcheck <executable> is useful if the problem exists on the GPU, else use valgrind <executable> **/
    cudaDeviceReset();
}
