//#include "precision.h"
//#include "UtilsDevice.cuh"
//#define MAXINT 0x7fffffff        

/*----------------------------------------------------------------------------------------
 *
 * This version is assumed to use only one block of threads.
 * The size of the arrays Key and Val is equal to TILE_SIZE defined below, while the
 * actual length of the Key-Val pair is nn.
 *
----------------------------------------------------------------------------------------*/
#include <nsp.h>
#include <nsparse_asm.h> 

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>
using namespace cub;

#define BLOCK_THREADS 512
#undef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 1

__device__ void mySort_KeyVal_row_1(int nn, int *Key, double *Val, int begin_bit, int end_bit,
                                char *sh_mem) {

   // Define the tile size
   enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };

   // Per-thread tile items
   int K_items[ITEMS_PER_THREAD];
   double V_items[ITEMS_PER_THREAD];

   // Specialize BlockRadixSort type for the present thread block
   typedef BlockRadixSort<int,BLOCK_THREADS,ITEMS_PER_THREAD,double> BlockRadixSortT;

   // Accomodate temporary storage in shared memory for sorting
   BlockRadixSortT::TempStorage *tmp_sort = (BlockRadixSortT::TempStorage*) sh_mem;

   // Load the keys and values into the buffer with valid_items = nn and setting the final block buffer to MAXINT
   LoadDirectBlocked(threadIdx.x, Key, K_items, nn, MAXINT);   
   LoadDirectBlocked(threadIdx.x, Val, V_items, nn, MAXINT);   
   
   // Barrier for smem reuse
   __syncthreads();

   // Sort keys
   BlockRadixSortT(*tmp_sort).SortBlockedToStriped(K_items,V_items,begin_bit,end_bit);

   // Store output in striped fashion
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Key, K_items,nn);
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Val, V_items,nn);

}

#undef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 2

__device__ void mySort_KeyVal_row_2(int nn, int *Key, double *Val, int begin_bit, int end_bit,
                                char *sh_mem) {

   // Define the tile size
   enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };

   // Per-thread tile items
   int K_items[ITEMS_PER_THREAD];
   double V_items[ITEMS_PER_THREAD];

   // Specialize BlockRadixSort type for the present thread block
   typedef BlockRadixSort<int,BLOCK_THREADS,ITEMS_PER_THREAD,double> BlockRadixSortT;

   // Accomodate temporary storage in shared memory for sorting
   BlockRadixSortT::TempStorage *tmp_sort = (BlockRadixSortT::TempStorage*) sh_mem;

   // Load the keys and values into the buffer with valid_items = nn and setting the final block buffer to MAXINT
   LoadDirectBlocked(threadIdx.x, Key, K_items, nn, MAXINT);   
   LoadDirectBlocked(threadIdx.x, Val, V_items, nn, MAXINT);   
   
   // Barrier for smem reuse
   __syncthreads();

   // Sort keys
   BlockRadixSortT(*tmp_sort).SortBlockedToStriped(K_items,V_items,begin_bit,end_bit);

   // Store output in striped fashion
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Key, K_items,nn);
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Val, V_items,nn);

}


#undef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 3

__device__ void mySort_KeyVal_row_3(int nn, int *Key, double *Val, int begin_bit, int end_bit,
                                char *sh_mem) {

   // Define the tile size
   enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };

   // Per-thread tile items
   int K_items[ITEMS_PER_THREAD];
   double V_items[ITEMS_PER_THREAD];

   // Specialize BlockRadixSort type for the present thread block
   typedef BlockRadixSort<int,BLOCK_THREADS,ITEMS_PER_THREAD,double> BlockRadixSortT;

   // Accomodate temporary storage in shared memory for sorting
   BlockRadixSortT::TempStorage *tmp_sort = (BlockRadixSortT::TempStorage*) sh_mem;

   // Load the keys and values into the buffer with valid_items = nn and setting the final block buffer to MAXINT
   LoadDirectBlocked(threadIdx.x, Key, K_items, nn, MAXINT);   
   LoadDirectBlocked(threadIdx.x, Val, V_items, nn, MAXINT);   
   
   // Barrier for smem reuse
   __syncthreads();

   // Sort keys
   BlockRadixSortT(*tmp_sort).SortBlockedToStriped(K_items,V_items,begin_bit,end_bit);

   // Store output in striped fashion
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Key, K_items,nn);
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Val, V_items,nn);

}


#undef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 4

__device__ void mySort_KeyVal_row_4(int nn, int *Key, double *Val, int begin_bit, int end_bit,
                                char *sh_mem) {

   // Define the tile size
   enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };

   // Per-thread tile items
   int K_items[ITEMS_PER_THREAD];
   double V_items[ITEMS_PER_THREAD];

   // Specialize BlockRadixSort type for the present thread block
   typedef BlockRadixSort<int,BLOCK_THREADS,ITEMS_PER_THREAD,double> BlockRadixSortT;

   // Accomodate temporary storage in shared memory for sorting
   BlockRadixSortT::TempStorage *tmp_sort = (BlockRadixSortT::TempStorage*) sh_mem;

   // Load the keys and values into the buffer with valid_items = nn and setting the final block buffer to MAXINT
   LoadDirectBlocked(threadIdx.x, Key, K_items, nn, MAXINT);   
   LoadDirectBlocked(threadIdx.x, Val, V_items, nn, MAXINT);   
   
   // Barrier for smem reuse
   __syncthreads();

   // Sort keys
   BlockRadixSortT(*tmp_sort).SortBlockedToStriped(K_items,V_items,begin_bit,end_bit);

   // Store output in striped fashion
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Key, K_items,nn);
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Val, V_items,nn);

}


#undef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 5

__device__ void mySort_KeyVal_row_5(int nn, int *Key, double *Val, int begin_bit, int end_bit,
                                char *sh_mem) {

   // Define the tile size
   enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };

   // Per-thread tile items
   int K_items[ITEMS_PER_THREAD];
   double V_items[ITEMS_PER_THREAD];

   // Specialize BlockRadixSort type for the present thread block
   typedef BlockRadixSort<int,BLOCK_THREADS,ITEMS_PER_THREAD,double> BlockRadixSortT;

   // Accomodate temporary storage in shared memory for sorting
   BlockRadixSortT::TempStorage *tmp_sort = (BlockRadixSortT::TempStorage*) sh_mem;

   // Load the keys and values into the buffer with valid_items = nn and setting the final block buffer to MAXINT
   LoadDirectBlocked(threadIdx.x, Key, K_items, nn, MAXINT);   
   LoadDirectBlocked(threadIdx.x, Val, V_items, nn, MAXINT);   
   
   // Barrier for smem reuse
   __syncthreads();

   // Sort keys
   BlockRadixSortT(*tmp_sort).SortBlockedToStriped(K_items,V_items,begin_bit,end_bit);

   // Store output in striped fashion
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Key, K_items,nn);
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Val, V_items,nn);

}


#undef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 6

__device__ void mySort_KeyVal_row_6(int nn, int *Key, double *Val, int begin_bit, int end_bit,
                                char *sh_mem) {

   // Define the tile size
   enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };

   // Per-thread tile items
   int K_items[ITEMS_PER_THREAD];
   double V_items[ITEMS_PER_THREAD];

   // Specialize BlockRadixSort type for the present thread block
   typedef BlockRadixSort<int,BLOCK_THREADS,ITEMS_PER_THREAD,double> BlockRadixSortT;

   // Accomodate temporary storage in shared memory for sorting
   BlockRadixSortT::TempStorage *tmp_sort = (BlockRadixSortT::TempStorage*) sh_mem;

   // Load the keys and values into the buffer with valid_items = nn and setting the final block buffer to MAXINT
   LoadDirectBlocked(threadIdx.x, Key, K_items, nn, MAXINT);   
   LoadDirectBlocked(threadIdx.x, Val, V_items, nn, MAXINT);   
   
   // Barrier for smem reuse
   __syncthreads();

   // Sort keys
   BlockRadixSortT(*tmp_sort).SortBlockedToStriped(K_items,V_items,begin_bit,end_bit);

   // Store output in striped fashion
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Key, K_items,nn);
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Val, V_items,nn);

}

#undef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 7

__device__ void mySort_KeyVal_row_7(int nn, int *Key, double *Val, int begin_bit, int end_bit,
                                char *sh_mem) {

   // Define the tile size
   enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };

   // Per-thread tile items
   int K_items[ITEMS_PER_THREAD];
   double V_items[ITEMS_PER_THREAD];

   // Specialize BlockRadixSort type for the present thread block
   typedef BlockRadixSort<int,BLOCK_THREADS,ITEMS_PER_THREAD,double> BlockRadixSortT;

   // Accomodate temporary storage in shared memory for sorting
   BlockRadixSortT::TempStorage *tmp_sort = (BlockRadixSortT::TempStorage*) sh_mem;

   // Load the keys and values into the buffer with valid_items = nn and setting the final block buffer to MAXINT
   LoadDirectBlocked(threadIdx.x, Key, K_items, nn, MAXINT);   
   LoadDirectBlocked(threadIdx.x, Val, V_items, nn, MAXINT);   
   
   // Barrier for smem reuse
   __syncthreads();

   // Sort keys
   BlockRadixSortT(*tmp_sort).SortBlockedToStriped(K_items,V_items,begin_bit,end_bit);

   // Store output in striped fashion
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Key, K_items,nn);
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Val, V_items,nn);

}


#undef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 8
__device__ void mySort_KeyVal_row_8(int nn, int *Key, double *Val, int begin_bit, int end_bit,
                                char *sh_mem) {

   // Define the tile size
   enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };

   // Per-thread tile items
   int K_items[ITEMS_PER_THREAD];
   double V_items[ITEMS_PER_THREAD];

   // Specialize BlockRadixSort type for the present thread block
   typedef BlockRadixSort<int,BLOCK_THREADS,ITEMS_PER_THREAD,double> BlockRadixSortT;

   // Accomodate temporary storage in shared memory for sorting
   BlockRadixSortT::TempStorage *tmp_sort = (BlockRadixSortT::TempStorage*) sh_mem;

   // Load the keys and values into the buffer with valid_items = nn and setting the final block buffer to MAXINT
   LoadDirectBlocked(threadIdx.x, Key, K_items, nn, MAXINT);   
   LoadDirectBlocked(threadIdx.x, Val, V_items, nn, MAXINT);   
   
   // Barrier for smem reuse
   __syncthreads();

   // Sort keys
   BlockRadixSortT(*tmp_sort).SortBlockedToStriped(K_items,V_items,begin_bit,end_bit);

   // Store output in striped fashion
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Key, K_items,nn);
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Val, V_items,nn);

}


#undef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 9
__device__ void mySort_KeyVal_row_9(int nn, int *Key, double *Val, int begin_bit, int end_bit,
                                char *sh_mem) {

   // Define the tile size
   enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };

   // Per-thread tile items
   int K_items[ITEMS_PER_THREAD];
   double V_items[ITEMS_PER_THREAD];

   // Specialize BlockRadixSort type for the present thread block
   typedef BlockRadixSort<int,BLOCK_THREADS,ITEMS_PER_THREAD,double> BlockRadixSortT;

   // Accomodate temporary storage in shared memory for sorting
   BlockRadixSortT::TempStorage *tmp_sort = (BlockRadixSortT::TempStorage*) sh_mem;

   // Load the keys and values into the buffer with valid_items = nn and setting the final block buffer to MAXINT
   LoadDirectBlocked(threadIdx.x, Key, K_items, nn, MAXINT);   
   LoadDirectBlocked(threadIdx.x, Val, V_items, nn, MAXINT);   
   
   // Barrier for smem reuse
   __syncthreads();

   // Sort keys
   BlockRadixSortT(*tmp_sort).SortBlockedToStriped(K_items,V_items,begin_bit,end_bit);

   // Store output in striped fashion
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Key, K_items,nn);
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Val, V_items,nn);

}

#undef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 10
__device__ void mySort_KeyVal_row_10(int nn, int *Key, double *Val, int begin_bit, int end_bit,
                                char *sh_mem) {

   // Define the tile size
   enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };

   // Per-thread tile items
   int K_items[ITEMS_PER_THREAD];
   double V_items[ITEMS_PER_THREAD];

   // Specialize BlockRadixSort type for the present thread block
   typedef BlockRadixSort<int,BLOCK_THREADS,ITEMS_PER_THREAD,double> BlockRadixSortT;

   // Accomodate temporary storage in shared memory for sorting
   BlockRadixSortT::TempStorage *tmp_sort = (BlockRadixSortT::TempStorage*) sh_mem;

   // Load the keys and values into the buffer with valid_items = nn and setting the final block buffer to MAXINT
   LoadDirectBlocked(threadIdx.x, Key, K_items, nn, MAXINT);   
   LoadDirectBlocked(threadIdx.x, Val, V_items, nn, MAXINT);   
   
   // Barrier for smem reuse
   __syncthreads();

   // Sort keys
   BlockRadixSortT(*tmp_sort).SortBlockedToStriped(K_items,V_items,begin_bit,end_bit);

   // Store output in striped fashion
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Key, K_items,nn);
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Val, V_items,nn);

}

#undef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 11
__device__ void mySort_KeyVal_row_11(int nn, int *Key, double *Val, int begin_bit, int end_bit,
                                char *sh_mem) {

   // Define the tile size
   enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };

   // Per-thread tile items
   int K_items[ITEMS_PER_THREAD];
   double V_items[ITEMS_PER_THREAD];

   // Specialize BlockRadixSort type for the present thread block
   typedef BlockRadixSort<int,BLOCK_THREADS,ITEMS_PER_THREAD,double> BlockRadixSortT;

   // Accomodate temporary storage in shared memory for sorting
   BlockRadixSortT::TempStorage *tmp_sort = (BlockRadixSortT::TempStorage*) sh_mem;

   // Load the keys and values into the buffer with valid_items = nn and setting the final block buffer to MAXINT
   LoadDirectBlocked(threadIdx.x, Key, K_items, nn, MAXINT);   
   LoadDirectBlocked(threadIdx.x, Val, V_items, nn, MAXINT);   
   
   // Barrier for smem reuse
   __syncthreads();

   // Sort keys
   BlockRadixSortT(*tmp_sort).SortBlockedToStriped(K_items,V_items,begin_bit,end_bit);

   // Store output in striped fashion
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Key, K_items,nn);
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Val, V_items,nn);

}

#undef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 12
__device__ void mySort_KeyVal_row_12(int nn, int *Key, double *Val, int begin_bit, int end_bit,
                                char *sh_mem) {

   // Define the tile size
   enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };

   // Per-thread tile items
   int K_items[ITEMS_PER_THREAD];
   double V_items[ITEMS_PER_THREAD];

   // Specialize BlockRadixSort type for the present thread block
   typedef BlockRadixSort<int,BLOCK_THREADS,ITEMS_PER_THREAD,double> BlockRadixSortT;

   // Accomodate temporary storage in shared memory for sorting
   BlockRadixSortT::TempStorage *tmp_sort = (BlockRadixSortT::TempStorage*) sh_mem;

   // Load the keys and values into the buffer with valid_items = nn and setting the final block buffer to MAXINT
   LoadDirectBlocked(threadIdx.x, Key, K_items, nn, MAXINT);   
   LoadDirectBlocked(threadIdx.x, Val, V_items, nn, MAXINT);   
   
   // Barrier for smem reuse
   __syncthreads();

   // Sort keys
   BlockRadixSortT(*tmp_sort).SortBlockedToStriped(K_items,V_items,begin_bit,end_bit);

   // Store output in striped fashion
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Key, K_items,nn);
   StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, Val, V_items,nn);

}

/*----------------------------------------------------------------------------------------
 *
 * Entry point for mySort_KeyVal_row
 *
----------------------------------------------------------------------------------------*/
__device__ void mySort_KeyVal_row(int nn, int end_bit, int *Key, double *Val,void *sh_mem) {  

   // Select the proper function
   switch ((nn-1)/BLOCK_THREADS + 1){

      case 1:
         mySort_KeyVal_row_1(nn,Key,Val,0,end_bit,(char*)sh_mem);
         break;

      case 2:
         mySort_KeyVal_row_2(nn,Key,Val,0,end_bit,(char*)sh_mem);
         break;

      case 3:
         mySort_KeyVal_row_3(nn,Key,Val,0,end_bit,(char*)sh_mem);
         break;

      case 4:
         mySort_KeyVal_row_4(nn,Key,Val,0,end_bit,(char*)sh_mem);
         break;

      case 5:
         mySort_KeyVal_row_5(nn,Key,Val,0,end_bit,(char*)sh_mem);
         break;

      case 6:
         mySort_KeyVal_row_6(nn,Key,Val,0,end_bit,(char*)sh_mem);
         break;

      case 7:
         mySort_KeyVal_row_7(nn,Key,Val,0,end_bit,(char*)sh_mem);
         break;

      case 8:
         mySort_KeyVal_row_8(nn,Key,Val,0,end_bit,(char*)sh_mem);
         break;

      case 9:
         mySort_KeyVal_row_9(nn,Key,Val,0,end_bit,(char*)sh_mem);
         break;

      case 10:
         mySort_KeyVal_row_10(nn,Key,Val,0,end_bit,(char*)sh_mem);
         break;

      case 11:
         mySort_KeyVal_row_11(nn,Key,Val,0,end_bit,(char*)sh_mem);
         break;

      case 12:
         mySort_KeyVal_row_12(nn,Key,Val,0,end_bit,(char*)sh_mem);
         break;

      default:
         if (nn > blockDim.x && threadIdx.x == 0)
            printf("Vettore troppo grande %d > BLKSZ %d; case = %d, BLOCK_THREADS = %d\n",nn,blockDim.x,(nn-1)/BLOCK_THREADS + 1,BLOCK_THREADS);
         break;

   }

}

//----------------------------------------------------------------------------------------
__global__ void nsp_calc_val_sort_rows(int nSigBits, int *row_perm,int bin_offset,
                                         int *iat, int *ja, double *coef){

   extern __shared__ char sh_mem[];

   // Retrieve thread infos
   int rid = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];
   int offset;
   int nz;

   // offset and nonzero of the row
   offset = iat[rid];
   nz = iat[rid + 1] - offset;

   // sort the rows
   mySort_KeyVal_row(nz,nSigBits,(int*) &(ja[offset]),&(coef[offset]),sh_mem);

}

#undef BLOCK_THREADS