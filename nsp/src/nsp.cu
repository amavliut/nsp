#include <iostream>
using namespace std;

#include <nsp.h>
#include <nsparse_asm.h> 

// Counts how many significant bits are necessary to represent nn
inline int countBITS(int nn){

   int k = 1;
   int imax = 2;
   while (nn >= imax){
      imax *= 2;
      k++;
   }
   return k;

}

//////////////////////////////////////////////////////////////////////////////////////////


template <typename ind_type, typename ind_type2>
__device__ __forceinline__ void hashmap_symbolic_bit(ind_type &nz, ind_type *check, ind_type2 key, const int SH_ROW_1){

  int hash = (key * HASH_SCAL) & SH_ROW_1;
  if (check[hash] != key) {
     while(check[hash]!=key && check[hash]!=-1) hash = (hash + 1) & SH_ROW_1;
     if (check[hash] != key) {
        while(1){
           ind_type old = atomicCAS(check + hash, -1, key);
           if (old == -1 || old == key) {
              nz += (unsigned int)old >> 31;
              break;
           }else hash = (hash + 1) & SH_ROW_1;
        }
     }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////

template <typename ind_type, typename ind_type2>
__device__ __forceinline__ void hashmap_symbolic_mod(ind_type &nz, ind_type *check, ind_type2 key, const int SH_ROW){

  int hash = (key * HASH_SCAL) % SH_ROW;
  if (check[hash] != key) {
     while(check[hash]!=key && check[hash]!=-1) hash = (hash + 1) % SH_ROW;
     if (check[hash] != key) {
        while(1){
           ind_type old = atomicCAS(check + hash, -1, key);
           if (old == -1 || old == key) {
              nz += (unsigned int)old >> 31;
              break;
           }else hash = (hash + 1) % SH_ROW;
        }
     }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////


 template <typename ind_type, typename ind_type2, typename val_type>
__device__ __forceinline__ void hashmap_bit(ind_type *check, val_type *value, ind_type2 key, val_type val,const int SH_ROW_1){

  // int hash = (key * HASH_SCAL) & SH_ROW_1;
  // if (check[hash] != key) {
  //    while(1){
  //       ind_type old = myatomicCAS(check + hash, -1, key);
  //       if (old == -1 || old == key) break;
  //       else hash = (hash + 1) & (SH_ROW - 1);
  //    }
  // }
  // atomicAdd_block(value + hash, val);


  int hash = (key * HASH_SCAL) & SH_ROW_1;
  if (check[hash] != key) {
     while(check[hash]!=key && check[hash]!=-1) hash = (hash + 1) & SH_ROW_1;
     if (check[hash] != key) {
        while(1){
           ind_type old = atomicCAS(check + hash, -1, key);
           if (old == -1 || old == key) break;
           else hash = (hash + 1) & SH_ROW_1;
        }
     }
  }
  atomicAdd_block(value + hash, val);
}

//////////////////////////////////////////////////////////////////////////////////////////

template <typename ind_type, typename ind_type2, typename val_type>
 __device__ __forceinline__ void hashmap_mod_count(ind_type *sh_sums, ind_type *check, val_type *value, ind_type2 key, val_type val,const int SH_ROW){


   int hash = (key * HASH_SCAL) % SH_ROW;
   if (check[hash] != key) {
      while(check[hash]!=key && check[hash]!=-1) hash = (hash + 1) % SH_ROW;
      if (check[hash] != key) {
         while(1){
            ind_type old = atomicCAS(check + hash, -1, key);
            if (old == -1 || old == key) {
               if (old == -1) atomicAdd_block(sh_sums,1);
               break;
            }
            else hash = (hash + 1) % SH_ROW;
         }
      }
   }
   atomicAdd_block(value + hash, val);
}

//////////////////////////////////////////////////////////////////////////////////////////

template <const int BS,const int nn, const int wnum>
__device__ __forceinline__ void dev_compactVal_chunk_dense(const unsigned char *check,const double *value,
                              int *sh_sums,int istrB,double *coef, int *ja,const int tid,const int wid){

   // First thread inits the first total running sum to zero
   if (threadIdx.x == 0) sh_sums[WARPSIZE] = 0;

   // Loop over all elements using 1 thread per element
   for (uint16_t jj = threadIdx.x; jj < nn; jj += BS){

      // Record "key" and decide whether it will deserve or not a position
      unsigned char key = check[jj];
      int           pos = (key == 0x00) ? 0:1; // could be int or uint16_t
     
      // Compute prefix sum in each warp
      #pragma unroll
      for (int i = 1; i < WARPSIZE; i <<= 1){
         int locSum = __shfl_up_sync(MASKFULL,pos,i,WARPSIZE);
         if (tid >= i) pos += locSum;
      }

      // Write the sum of the warp into the sh_sums array
      if (tid == WARPSIZE-1) sh_sums[wid] = pos;
      __syncthreads();

      // First warp computes the blockSum
      if (wid == 0){
         int warpSum = (tid < wnum) ? sh_sums[tid]:0;
         #pragma unroll
         for (int i = 1; i < WARPSIZE; i <<= 1){
            int locSum = __shfl_up_sync(MASKFULL,warpSum,i,WARPSIZE);
            if (tid >= i) warpSum += locSum;
         }
         sh_sums[tid] = warpSum;
      }
      __syncthreads();

      // add warp and block running sums
      pos += (wid > 0) ? sh_sums[WARPSIZE] + sh_sums[wid-1] : sh_sums[WARPSIZE];

      // Store key_value back in Key
      if (key != 0x00){
           ja[pos-1] = jj + istrB;
         coef[pos-1] = value[jj];
      }
      __syncthreads();

      // Store the new total running sum value
      if (threadIdx.x == BS-1) sh_sums[WARPSIZE] = pos;
   }

}

//////////////////////////////////////////////////////////////////////////////////////////


static __device__ __forceinline__ void dev_compactKeyVal_inplace(const int nn, int *nc, int *Key, double *Data,
                                  int *sh_sums,int col_offset){

   // Thread and warp info
   int id = threadIdx.x;
   int lane_id = id%warpSize;
   int warp_id = id/warpSize;
   int blksz = blockDim.x;
   int nwarps_in_blk = (blksz + warpSize - 1) / warpSize;

   // Other variables
   int i,jj;
   int pos,key_value,locSum,warpSum,blockSum,last_id;
   double data_value;

   // First thread inits the first total running sum to zero
   if (id == 0) sh_sums[warpSize] = 0;

   // Loop over all elements using 1 thread per element
   for (jj = id; jj < nn; jj += blksz){

      // Record "key_value" and decide wether it will deserve or not a position
      key_value = Key[jj];
      data_value = Data[jj];
      pos = (key_value<0) ? 0:1;
     
      // Compute prefix sum in each warp
      #pragma unroll
      for (i = 1; i < warpSize; i *= 2){
          locSum = __shfl_up_sync(MASKFULL,pos, i, warpSize);
          if (lane_id >= i) pos += locSum;
      }

      // Write the sum of the warp into the sh_sums array
      if (lane_id == warpSize-1) sh_sums[warp_id] = pos;
      __syncthreads();

      // First warp computes the blockSum
      if (warp_id == 0){
         warpSum = (lane_id < nwarps_in_blk) ? sh_sums[lane_id]:0;
         #pragma unroll
         for (i = 1; i < warpSize; i *= 2){
            locSum = __shfl_up_sync(MASKFULL,warpSum,i,warpSize);
            if (lane_id >= i) warpSum += locSum;
         }
         sh_sums[lane_id] = warpSum;
      }
      __syncthreads();

      // Store in blockSum the running sum correspondig to this warp
      blockSum = (warp_id > 0) ? sh_sums[warp_id-1]:0;
      // Add blockSum and the total running sum to determine the position of each entry
      pos += blockSum + sh_sums[warpSize];

      // Store key_value back in Key
      if (key_value >= 0){
         Key[pos-1] = key_value + col_offset;
         Data[pos-1] = data_value;
      }
      __syncthreads();

      // Store the new total running sum value
      if (id == blksz-1) sh_sums[warpSize] = pos;
   }

   // Last active thread in the last cycle dumps the final number of entries
   last_id = nn%blksz;
   last_id = (!last_id) ? blksz-1:last_id-1;
   if (id == last_id) *nc = pos;

}


//////////////////////////////////////////////////////////////////////////////////////////

void nsp_init_bin(sfBIN *bin, const int nrows_C) {

   // allocating streams
   bin->stream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * BIN_NUM);
   for (int i = 0; i < BIN_NUM; i++) {
      cudaStreamCreate(&(bin->stream[i]));
   }

   // allocate host members
   //if ( bin->stream == NULL ) throw linsol_error("nsp_init_bin","stream");
   if ( bin->stream == NULL ){ 
        printf("ERROR: bin->stream is NULL\n");
   }

   bin->h_bin_size = (int *)malloc(sizeof(int) * BIN_NUM);
   bin->h_bin_offset = (int *)malloc(sizeof(int) * BIN_NUM);

   // allocate device members
   checkCudaErrors(cudaMalloc((void **)&(bin->d_row_nz), sizeof(int) * (nrows_C + 1)));
   checkCudaErrors(cudaMalloc((void **)&(bin->d_max), sizeof(int)));
   checkCudaErrors(cudaMalloc((void **)&(bin->d_bin_size), sizeof(int) * BIN_NUM));
   checkCudaErrors(cudaMalloc((void **)&(bin->d_bin_offset), sizeof(int) * BIN_NUM));
   checkCudaErrors(cudaMalloc((void **)&(bin->d_row_perm), sizeof(int) * nrows_C));

   // set shared memory infos
   bin->SHTB_cmp_max = SHTB / 12;
   bin->SHTB_set_max = bin->SHTB_cmp_max * 2; 
   bin->IMB_MIN = bin->SHTB_set_max / 16;
   bin->B_MIN   = bin->SHTB_cmp_max / 16;

}

//////////////////////////////////////////////////////////////////////////////////////////


void nsp_release_bin(sfBIN *bin) {
   // destroy streams
   for (int i = 0; i < BIN_NUM; i++) {
       cudaStreamDestroy(bin->stream[i]);
   }
   free(bin->stream);
   free(bin->h_bin_size);
   free(bin->h_bin_offset);
   
   // free device members
   cudaFree(bin->d_max);
   cudaFree(bin->d_row_nz);
   cudaFree(bin->d_row_perm);
   cudaFree(bin->d_bin_size);
   cudaFree(bin->d_bin_offset);
}



//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////


__global__ void nsp_set_intprod_num(int *iat_A, int *ja_A, int *iat_B,
                                    int *row_intprod, int nrows_C) {
   // retrieve row index
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= nrows_C) return;
   // initialize number of intermediate products
   int nz_per_row = 0;
   // compute number of intprod
   for (int j = iat_A[i]; j < iat_A[i+1]; j++) {
      int jcol_A = ja_A[j];
      nz_per_row += iat_B[jcol_A+1] - iat_B[jcol_A];
   }
   // store the number
   row_intprod[i] = nz_per_row;
}

//////////////////////////////////////////////////////////////////////////////////////////


__global__ void nsp_set_bin(int *row_nz, int *bin_size, int nrows_C) {

   // retrieve row index
   int rid = blockIdx.x * blockDim.x + threadIdx.x;

   if (rid >= nrows_C) return;

   // registers
   int nz_per_row = row_nz[rid];
   int loc_bin[BIN_NUM] = {0};

   if      (nz_per_row <= 64)          loc_bin[0]++;            // pwarp 
   else if (nz_per_row <= 128)         loc_bin[1]++;            // pwarp 
   else if (nz_per_row <= 256)         loc_bin[2]++;            // pwarp
   else if (nz_per_row <= 512)         loc_bin[3]++;            // pwarp
   else if (nz_per_row <= 1024)        loc_bin[4]++;            // tb
   else if (nz_per_row <= 2048)        loc_bin[5]++;            // tb
   else if (nz_per_row <= 4096)        loc_bin[6]++;            // tb
   else if (nz_per_row <= 8192)        loc_bin[7]++;            // tb
   else if (nz_per_row <= 12288)       loc_bin[8]++;            // tb
   else                                loc_bin[9]++;            // chunk
   
   #pragma unroll
   for(int i=0;i<BIN_NUM-1;i++){
      atomicAdd(bin_size+i, loc_bin[i]);
   }

}
//////////////////////////////////////////////////////////////////////////////////////////


__global__ void nsp_set_row_perm(int *bin_size, int *bin_offset,
                                 int *max_row_nz, int *row_perm,
                                 int nrows_C) {

   // retrieve row index
   int i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i >= nrows_C) return;

   // other registers
   int nz_per_row = max_row_nz[i];
   int dest;

   // BINNUM = 10
   if (nz_per_row <= 64){                    // pwarp 
      dest = atomicAdd(bin_size, 1);
      row_perm[bin_offset[0] + dest] = i;
   }
   else if (nz_per_row <= 128){              // pwarp 
      dest = atomicAdd(bin_size+1, 1);
      row_perm[bin_offset[1] + dest] = i;
   }
   else if (nz_per_row <= 256){              // pwarp
      dest = atomicAdd(bin_size+2, 1);
      row_perm[bin_offset[2] + dest] = i;
   }
   else if (nz_per_row <= 512){              // pwarp
      dest = atomicAdd(bin_size+3, 1);
      row_perm[bin_offset[3] + dest] = i;
   }
   else if (nz_per_row <= 1024){             // tb
      dest = atomicAdd(bin_size+4, 1);
      row_perm[bin_offset[4] + dest] = i;
   }
   else if (nz_per_row <= 2048){             // tb
      dest = atomicAdd(bin_size+5, 1);
      row_perm[bin_offset[5] + dest] = i;
   }
   else if (nz_per_row <= 4096){             // tb
      dest = atomicAdd(bin_size+6, 1);
      row_perm[bin_offset[6] + dest] = i;
   }
   else if (nz_per_row <= 8192){             // tb
      dest = atomicAdd(bin_size+7, 1);
      row_perm[bin_offset[7] + dest] = i;
   }
   else if (nz_per_row <= 12288){            // tb
      dest = atomicAdd(bin_size+8, 1);
      row_perm[bin_offset[8] + dest] = i;
   }
   else{                                     // large
      dest = atomicAdd(bin_size+9, 1);
      row_perm[bin_offset[9] + dest] = i;
   }

}

//////////////////////////////////////////////////////////////////////////////////////////


// Estimate size of C rows and set-up sfBIN
void nsp_set_max_bin( int *d_iat_A, int *d_ja_A, int *d_iat_B, sfBIN *bin, int nrows_C, int &DIRECT) {

   // set handles
   int *h_bin_offset = bin->h_bin_offset;
   int *h_bin_size   = bin->h_bin_size;
   int *d_row_nz     = bin->d_row_nz;
   int *d_bin_offset = bin->d_bin_offset;
   int *d_bin_size   = bin->d_bin_size;
   int *d_row_perm   = bin->d_row_perm;

   // initialize sfBIN structure to 0
   for (int i = 0; i < BIN_NUM; i++) {
        h_bin_size[i] = 0;
      h_bin_offset[i] = 0;
   }
   cudaMemset(d_bin_size, 0, BIN_NUM * sizeof(int));

   // estimate size of C rows as number of intprod
   int BS = BLKSIZE_MxM;
   int GS = div_round_up(nrows_C,BS);

   nsp_set_intprod_num<<<GS,BS>>> (d_iat_A, d_ja_A, d_iat_B, d_row_nz, nrows_C);
   nsp_set_bin<<<GS,BS>>> (d_row_nz, d_bin_size, nrows_C);
   
   // copy group sizes from Device to Host
   cudaMemcpy(h_bin_size, d_bin_size, sizeof(int) * BIN_NUM, cudaMemcpyDeviceToHost);
   // if the largest bin is dominant (has > 15% of the rows) then don't permute the rows and use direct access
   int i = BIN_NUM - 1;
   while (h_bin_size[i] == 0) i--;

   if ((float)h_bin_size[i]/nrows_C > 0.15 ) { // add condition that it is not the chunk bins
      // nulify the use of other bins
      for (int j = 0; j < i; j++) h_bin_size[j] = 0;
      h_bin_size[i] = nrows_C;      // set up the largest bin to process all the rows
      d_row_perm = NULL;         // nulify the row permutation pointer
      DIRECT = 1;
   }else{
      // reset to 0 group sizes on the Device (recomputed later in set_row_perm)
      cudaMemset(d_bin_size, 0, BIN_NUM * sizeof(int));
      // set-up host
      for (int i = 0; i < BIN_NUM - 1; i++) {
         h_bin_offset[i+1] = h_bin_offset[i] + int(h_bin_size[i]);
      }
      cudaMemcpy(d_bin_offset, h_bin_offset, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);
      nsp_set_row_perm<<<GS,BS>>>(d_bin_size,d_bin_offset,d_row_nz,d_row_perm,nrows_C);
   }
   // nulify the row_nz pointer to use atomic add in each_tb kernel
   if (i > 3) cudaMemset(d_row_nz, 0, nrows_C*sizeof(int));

   #if defined BENCHMARK
      for (int i = 0; i < BIN_NUM; i++) cout << h_bin_size[i] << " ";
      cout << endl;
   #endif

}


//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

template <const int pWARP,const int SH_ROW>
__global__ void nsp_set_row_nz_bin_mpwarp(int * __restrict__ iat_A, int * __restrict__ ja_A, 
                                         const int * __restrict__ iat_B, int * __restrict__ ja_B,
                                         int * __restrict__ row_perm, int * __restrict__ row_nz, 
                                         const int bin_offset,const int nrows) {

   // retrieve thread infos
   int mid  = (blockIdx.x * (blockDim.x / pWARP) + threadIdx.x / pWARP);
   int rid  = (row_perm == nullptr) ? mid : row_perm[mid + bin_offset];
   int tid  = threadIdx.x & (pWARP - 1);
   int wid  = threadIdx.x / pWARP;

   // registers
   int jr;
   int je,ke;
   int jcol_A;
   int key;
   int nz = 0;   // initialize number of non zeros

   // block shared memory
   extern __shared__ int sh_mem[];
   int *check = (int*)sh_mem + wid * SH_ROW;

   // initialize hash table
   #pragma unroll 
   for (jr = tid; jr < SH_ROW; jr += pWARP) {
      check[jr] = -1;
   }

   // warp synchronization to ensure check initialization
   __syncwarp();

   if (mid < nrows) {
      // loop over A-row coefficients
      for (je = iat_A[rid]; je < iat_A[rid + 1]; je++) {
         // load from global memory without using the cache
         jcol_A = load_glob(ja_A + je);
         // loop over B-row coefficients
         for (ke = iat_B[jcol_A]+tid; ke < iat_B[jcol_A + 1]; ke+=pWARP) {
            // load from global memory using the cache
            key = ja_B[ke];
            hashmap_symbolic_bit(nz, check, key, SH_ROW-1);
         } // end loop over B-row coefficients
      } // end loop over A-row coefficients
   }
   // pwarp reduction of nz
   __syncwarp(MASKFULL);
   #pragma unroll 
   for( jr = pWARP>>1; jr>0; jr>>=1) {
      nz += __shfl_down_sync( MASKFULL, nz, jr, pWARP );
   }
   // store the final value
   if (tid == 0 && mid < nrows) row_nz[rid] = nz;
}


//////////////////////////////////////////////////////////////////////////////////////////

template <const int SH_ROW>
__global__ void nsp_set_row_nz_bin_each_tb(int * __restrict__ iat_A, int * __restrict__ ja_A, 
                                           const int * __restrict__ iat_B,const int * __restrict__ ja_B,
                                           int * __restrict__ row_perm, int * __restrict__ row_nz, const int bin_offset) {

   // retrieve thread infos
   int rid  = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];
   int tid  = threadIdx.x & (WARPSIZE - 1);
   int wid  = threadIdx.x / WARPSIZE;
   int wnum = blockDim.x / WARPSIZE;

   // registers
   int jr;
   int je,ke;
   int nz = 0;
   int jcol_A;
   int key;

   // block shared memory
   extern __shared__ int sh_mem[];
   int *check = (int*) sh_mem;

   // initialize hash table
   #pragma unroll 
   for (jr = threadIdx.x; jr < SH_ROW; jr += blockDim.x) {
       check[jr] = -1;
   }

   // block synchronization to ensure check initialization
   __syncthreads();
   // loop over A-row coefficients
   for (je = iat_A[rid] + wid; je < iat_A[rid + 1]; je += wnum) {
      // load from global memory without using the cache
      jcol_A = load_glob(ja_A + je);
      // loop over B-row coefficients
      for (ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke += WARPSIZE) {
         // load from global memory using the cache
         key = ja_B[ke];
         hashmap_symbolic_bit(nz, check, key, SH_ROW-1);
      } // end loop over B-row coefficients
   } // end loop over A-row coefficients
   // warp reduction of nz
   __syncwarp(MASKFULL);
   #pragma unroll 
   for( jr = WARPSIZE>>1; jr>0; jr>>=1) {
      nz += __shfl_down_sync( MASKFULL, nz, jr, WARPSIZE );
   }
   // block reduction of nz using 1 thread for each warp
   // __syncthreads();
   if (tid == 0) atomicAdd(row_nz + rid, nz);

}



//////////////////////////////////////////////////////////////////////////////////////////

__global__ void nsp_set_row_nz_bin_each_tb_chunk(const int *iat_A,const int *ja_A,
                                                 const int * __restrict__ iat_B, const int * __restrict__ ja_B,
                                                 const int *row_perm, int *row_nz,
                                                 const int bin_offset,const int nrows_tb,
                                                 const int ncols_C,const int SH_ROW) {

   // retrieve thread infos
   int rid  = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];
   int tid  = threadIdx.x & (WARPSIZE - 1);
   int wid  = threadIdx.x / WARPSIZE;
   int wnum = blockDim.x / WARPSIZE;

   // registers
   int jr;
   int je,ke;
   int ichunk;
   int nz;
   int jcol_A,jcol_B;
   int istrB,iendB;
   int key,hash,old;

   // block shared memory
   extern __shared__ int sh_mem[];
   #if defined LARGE_NCOLS
      int *check = (int*) sh_mem;
   #else
      int *check = (int*) sh_mem;
   #endif

   // initialize hash table
   for (jr = threadIdx.x; jr < SH_ROW; jr += blockDim.x) {
       check[jr] = -1;
   }

   // initialize number of non zeros
   if (threadIdx.x == 0 ) row_nz[rid] = 0;

   // block synchronization to ensure check initialization
   __syncthreads();

   // start loop over B chunks
   ichunk = 0;
   istrB  = 0;
   iendB  = SH_ROW;
   while (1) {

      // initialize number of non-zeros for the chunk
      nz = 0;

      // loop over A-row coefficients
      for (je = iat_A[rid] + wid; je < iat_A[rid + 1]; je += wnum) {

         // load from global memory without using the cache
         jcol_A = load_glob(ja_A + je);

         // loop over row terms of B
         for (ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke += WARPSIZE) {

            // load from global memory using the cache
            jcol_B = ja_B[ke];

            // check end of the chunk
            if ( jcol_B >= iendB ) goto cycle_loop_A;

            // check column index
            if ( jcol_B >= istrB ) {

               key  = jcol_B - istrB;
               hash = (key * HASH_SCAL) & (SH_ROW - 1);

               #if defined HASH_UPD
               if (check[hash] != key) {
                  while(1){
                     old = atomicCAS(check + hash, -1, key);
                     if (old == -1 || old == key) {
                        if (old == -1) nz++;
                        break; 
                     }else hash = (hash + 1) & (SH_ROW - 1);
                  }
               }
               #elif defined HASH_NSPARSE
                  // put the key inside the hash table
                  if (check[hash] != key) {
                     while (1){
                        old = atomicCAS(check + hash, -1, key);
                        if (old == -1) {
                           nz++;
                           break;
                        } else {
                           if (old != key){ 
                              hash = (hash + 1) & (SH_ROW - 1);
                           } else {
                              break;
                           }
                        }
                     }
                  }
               #else // default hash algorithm:
                  // put the key inside the hash table
                  while (1) {
                     if (check[hash] == key) {
                        break;
                     } else if (check[hash] == -1) {
                        old = atomicCAS(check + hash, -1, key);
                        if (old == -1) {
                           nz++;
                           break;
                        }
                     } else if (check[hash] != key) {
                        hash = (hash + 1) & (SH_ROW - 1);
                     }
                  }
               #endif

            } // end check column index

         } // end loop over row terms of B

         cycle_loop_A: ;

      } // end loop over A-row coefficients

      // warp reduction of nz
      // __syncwarp(MASKFULL);
      for( jr = WARPSIZE/2; jr > 0; jr /= 2) {
         nz += __shfl_down_sync( MASKFULL, nz, jr, WARPSIZE );
      }

      // block reduction of nz using 1 thread for each warp
      __syncthreads();
      if (threadIdx.x == 0) check[0] = 0;
      __syncthreads();
      if (tid == 0) atomicAdd(check, nz);
      __syncthreads();

      // store the final value
      if (threadIdx.x == 0) row_nz[rid] += check[0];
      __syncthreads();

      // check end of loop over B chunks
      if ( iendB >= ncols_C ) break;

      ichunk++;

      // update B indeces
      istrB  = iendB;
      iendB += SH_ROW;

      // initialize shared scratch
      for (jr = threadIdx.x; jr < SH_ROW; jr += blockDim.x) {
          check[jr] = -1;
      }

      // synchronize before next chunk cycle
      __syncthreads();

   } // end loop over B chunks

}

//////////////////////////////////////////////////////////////////////////////////////////


__global__ void nsp_set_row_nz_bin_each_tb_large(const int *iat_A,const int *ja_A,
                                                 const int * __restrict__ iat_B,const int * __restrict__ ja_B,
                                                 const int *row_perm, int *row_nz,
                                                 int *fail_count, int *fail_perm,
                                                 const int bin_offset, const int nrows_tb,const int SH_ROW ){

   // retrieve thread infos
   int rid  = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];
   int tid  = threadIdx.x & (WARPSIZE - 1);
   int wid  = threadIdx.x / WARPSIZE;
   int wnum = blockDim.x / WARPSIZE;

   // registers
   int jr;
   int je,ke;
   int jcol_A;
   int key,hash,old;
   int count;
   int border; 
   int dest;

   // block shared memory
   extern __shared__ int sh_mem[];
   #if defined LARGE_NCOLS
      int *check = (int*) sh_mem;
   #else
      int *check = (int*) sh_mem;
   #endif
    __shared__ int snz[1];

   // initialize hash table
   for (jr = threadIdx.x; jr < SH_ROW; jr += blockDim.x) {
       check[jr] = -1;
   }
   if (threadIdx.x == 0) snz[0] = 0;

   // block synchronization to ensure check initialization
   __syncthreads();

   // initialize registers
   count = 0;
   border = SH_ROW >> 1;

   // loop over A-row coefficients
   for (je = iat_A[rid] + wid; je < iat_A[rid + 1]; je += wnum) {

      // load from global memory without using the cache
      jcol_A = load_glob(ja_A + je);

      // loop over B-row coefficients
      for (ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke += WARPSIZE) {

         // load from global memory using the cache
         key = ja_B[ke];

         hash = (key * HASH_SCAL) & (SH_ROW - 1);

         // put the key inside the hash table
         while (count < border && snz[0] < border) {
            
            if (check[hash] == key) {
               // key already added
               break; 
            } else if (check[hash] == -1) {
               // add the key
               old = atomicCAS(check + hash, -1, key);
               if (old == -1) {
                  atomicAdd(snz,1);
                  break;
               }
            } else if (check[hash] != key) {
               // find a free place to add the key
               hash = (hash + 1) & (SH_ROW - 1);
               count++;
            }
         }
   
         // check fail: gone outside hash
         if (count >= border || snz[0] >= border) break;

      } // end loop over B-row coefficients

      // check fail: gone outside hash
      if (count >= border || snz[0] >= border) break;

   } // end loop over A-row coefficients

   // block syncronization
   __syncthreads();

   // check compuatation fail
   if (count >= border || snz[0] >= border) {
      // store failed row index
      if (threadIdx.x == 0) {
         dest = atomicAdd(fail_count, 1);
         fail_perm[dest] = rid;
      }
   } else {
      // store row non-zeros
      if (threadIdx.x == 0) {
         row_nz[rid] = snz[0];
      }
   }

}

//////////////////////////////////////////////////////////////////////////////////////////

template <int SH_ROW>
__global__ void nsp_set_row_nz_bin_each_tb_max(int * __restrict__ iat_A, int * __restrict__ ja_A, 
                                           const int * __restrict__ iat_B,const int * __restrict__ ja_B,
                                           int * __restrict__ row_perm, int * __restrict__ row_nz, const int bin_offset) {

   // retrieve thread infos
   int rid  = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];
   int tid  = threadIdx.x & (WARPSIZE - 1);
   int wid  = threadIdx.x / WARPSIZE;
   int wnum = blockDim.x / WARPSIZE;

   // registers
   int jr;
   int je,ke;
   int nz = 0;
   int jcol_A;
   int key;

   // block shared memory
   extern __shared__ int sh_mem[];
   int *check = (int*) sh_mem;

   // initialize hash table
   #pragma unroll 
   for (jr = threadIdx.x; jr < SH_ROW; jr += blockDim.x) {
       check[jr] = -1;
   }

   // block synchronization to ensure check initialization
   __syncthreads();

   // loop over A-row coefficients
   for (je = iat_A[rid] + wid; je < iat_A[rid + 1]; je += wnum) {
      // load from global memory without using the cache
      jcol_A = load_glob(ja_A + je);
      // loop over B-row coefficients
      for (ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke += WARPSIZE) {
         // load from global memory using the cache
         key = ja_B[ke];
         hashmap_symbolic_mod(nz, check, key, SH_ROW-1);
      } // end loop over B-row coefficients
   } // end loop over A-row coefficients

   // warp reduction of nz
   __syncwarp(MASKFULL);
   #pragma unroll 
   for( jr = WARPSIZE>>1; jr>0; jr>>=1) {
      nz += __shfl_down_sync( MASKFULL, nz, jr, WARPSIZE );
   }
   // block reduction of nz using 1 thread for each warp
   // __syncthreads();
   if (tid == 0) atomicAdd(row_nz + rid, nz);
}


//////////////////////////////////////////////////////////////////////////////////////////


void nsp_set_row_nnz( int *d_iat_A, int *d_ja_A, int *d_iat_B, int *d_ja_B, int *d_iat_C,
                      sfBIN *bin, int nrows_C, int ncols_C, int *nterm_C, int DIRECT) {

   // set handles
   int *h_bin_offset   = bin->h_bin_offset;
   int *h_bin_size     = bin->h_bin_size;
   int *d_row_perm     = (DIRECT) ? nullptr : bin->d_row_perm;
   int *d_row_nz       = bin->d_row_nz;

   // define varibles for GPU resources
   size_t shmemsize;
   int GS,BS,SH;

   // cub exclusive scan
   void     *d_temp_storage = NULL;
   size_t   temp_storage_bytes = 0;
   cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_row_nz, d_iat_C, nrows_C+1);
   cudaMalloc(&d_temp_storage, temp_storage_bytes);

   // loop over groups
   for (int i = BIN_NUM - 1; i >= 0; i--) {
      // check sizes
      if (h_bin_size[i] > 0) {
         // select group kernel
         switch (i) {
            case 0:  // <= 64
               #if CC == 86
                  BS = 96;
               #else
                  BS = 64;
               #endif
               SH = 64 * BS / 8;
               shmemsize = SH * sizeof(int);
               nsp_set_row_nz_bin_mpwarp<8,64><<<h_bin_size[i]/(BS/8)+1, BS, shmemsize, bin->stream[i]>>>
                                       (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                        d_row_perm,d_row_nz,
                                        h_bin_offset[i],h_bin_size[i]);
               break;
            case 1:  // <= 128
               #if CC == 86
                  BS = 96;
               #else
                  BS = 64;
               #endif
               SH = 128 * BS / 16;
               shmemsize = SH * sizeof(int);
               nsp_set_row_nz_bin_mpwarp<16,128><<<h_bin_size[i]/(BS/16)+1, BS, shmemsize, bin->stream[i]>>>
                                       (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                        d_row_perm,d_row_nz,
                                        h_bin_offset[i],h_bin_size[i]);
               break;
            case 2 : // <= 256
               #if CC == 86
                  BS = 96;
               #else
                  BS = 64;
               #endif
               SH = 256 * BS / 32;
               shmemsize = SH * sizeof(int);
               nsp_set_row_nz_bin_mpwarp<32,256><<<h_bin_size[i]/(BS/32)+1, BS, shmemsize, bin->stream[i]>>>
                                       (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                        d_row_perm,d_row_nz,
                                        h_bin_offset[i],h_bin_size[i]);
               break;
            case 3 : // <= 512
               #if CC == 86
                  BS = 96;
               #else
                  BS = 64;
               #endif
               GS = h_bin_size[i];
               SH = 512;
               shmemsize = SH * sizeof(int);
               nsp_set_row_nz_bin_each_tb<512><<<GS, BS, shmemsize, bin->stream[i]>>>
                                         (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                          d_row_perm,d_row_nz,
                                          h_bin_offset[i]);
               break;
            case 4 : // <= 1024
               BS = 128;
               GS = h_bin_size[i];
               SH = 1024;
               shmemsize = SH * sizeof(int);
               nsp_set_row_nz_bin_each_tb<1024><<<GS, BS, shmemsize, bin->stream[i]>>>
                                         (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                          d_row_perm,d_row_nz,
                                          h_bin_offset[i]);
               break;
            case 5 : // <= 2048
               BS = 256;
               GS = h_bin_size[i];
               SH = 2048;
               shmemsize = SH * sizeof(int);
               nsp_set_row_nz_bin_each_tb<2048><<<GS, BS, shmemsize, bin->stream[i]>>>
                                         (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                          d_row_perm,d_row_nz,h_bin_offset[i]);
               break;
            case 6 : // <= 4096
               BS = 512;
               GS = h_bin_size[i];
               SH = 4096;
               shmemsize = SH * sizeof(int);
               nsp_set_row_nz_bin_each_tb<4096><<<GS, BS, shmemsize, bin->stream[i]>>>
                                         (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                          d_row_perm,d_row_nz,
                                          h_bin_offset[i]);
               break;
            case 7 : // <= 8192
               #if CC == 86
                  BS = 768;
               #else
                  BS = 1024;
               #endif
               GS = h_bin_size[i];
               SH = 8192;
               shmemsize = SH * sizeof(int);
               nsp_set_row_nz_bin_each_tb<8192><<<GS, BS, shmemsize, bin->stream[i]>>>
                                         (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                          d_row_perm,d_row_nz,
                                          h_bin_offset[i]);
               break;
            case 8 : // <= 12288
               #if CC == 86
                  BS = 768;
               #else
                  BS = 1024;
               #endif
               GS = h_bin_size[i];
               SH = 12288;
               shmemsize = SH * sizeof(int);
               nsp_set_row_nz_bin_each_tb_max<12288><<<GS, BS, shmemsize, bin->stream[i]>>>
                                         (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                          d_row_perm,d_row_nz,
                                          h_bin_offset[i]);
               break;

            case 9 :
               // start scope case 9 (to allocate inside a switch)
               {
                  // prepare auxiliary variables for large rows
                  int  h_fail_count = 0;
                  int *d_fail_count = NULL;
            	  checkCudaErrors(cudaMalloc((void **)&d_fail_count, sizeof(int)));
                  //cudaError_t cudaError = cudaMalloc((void **)&d_fail_count, sizeof(int));
                  //CheckCudaError("nsp_set_row_nnz","allocating d_fail_count",cudaError);

                  cudaMemcpy(d_fail_count, &h_fail_count, sizeof(int), cudaMemcpyHostToDevice);

                  int* d_vec_fail_perm;
				  checkCudaErrors(cudaMalloc((void **)&d_vec_fail_perm, sizeof(int) * h_bin_size[i]));

                  int *d_fail_perm = d_vec_fail_perm;
				  
                  // set GPU resources
                  #if CC == 86
                     BS = 768;
                  #else
                     BS = 1024;
                  #endif
                  GS = h_bin_size[i];
                  SH = bin->SHTB_set_max;
                  #if defined LARGE_NCOLS
                     shmemsize = SH * sizeof(int);
                  #else
                     shmemsize = SH * sizeof(int);
                  #endif
                  // try to compute all the rows using the standard kernel
                  nsp_set_row_nz_bin_each_tb_large<<<GS, BS, shmemsize, bin->stream[i]>>>
                                                  (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                                   d_row_perm,d_row_nz,
                                                   d_fail_count,d_fail_perm,
                                                   h_bin_offset[i],h_bin_size[i],SH);
 
                  // check if the computation failed for some rows
                  cudaMemcpy(&h_fail_count, d_fail_count, sizeof(int), cudaMemcpyDeviceToHost);
                  if (h_fail_count > 0) {
                     // compute rows using the chunk kernel
                     GS = h_fail_count;
                     nsp_set_row_nz_bin_each_tb_chunk<<<GS, BS, shmemsize, bin->stream[i]>>>
                                                           (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                                            d_fail_perm,d_row_nz,
                                                            0,h_fail_count,ncols_C,SH);
                  }
                  // remove auxiliary variables for large rows
                  cudaFree(d_fail_count);
                  //d_vec_fail_perm.resize(0);
				  cudaFree(d_vec_fail_perm);
               } // end scope case 9
               break;
            default :
			   printf("nsp_set_row_nnz -- kernel not implemented yet");
               //throw linsol_error ("nsp_set_row_nnz","kernel not implemented yet");          
               break;

         } // end select case
      } // end check group size
   } // end loop over groups

   // syncronize device
   cudaDeviceSynchronize();

   // Set row pointer of matrix C
   // thrust::exclusive_scan(thrust::device, d_row_nz, d_row_nz + (nrows_C + 1), d_iat_C, 0);
   cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_row_nz, d_iat_C, nrows_C + 1);

   cudaMemcpy(nterm_C, d_iat_C + nrows_C, sizeof(int), cudaMemcpyDeviceToHost);

   cudaFree(d_temp_storage);

}


//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////


__global__ void nsp_set_bin_min(int *row_nz, int *bin_size,
                            int nrows_C, int ncols_C) {

   // retrieve row index
   int rid = blockIdx.x * blockDim.x + threadIdx.x;

   int loc_bin[BIN_NUM] = {0};

   if (rid >= nrows_C) return;

   int nz_per_row = row_nz[rid];

   float density = (float)nz_per_row / ncols_C * 100.;
   if ( (ncols_C <= MAX_SH_DENSE && density > MIN_DENSITY) || (density > MIN_DENSITY_chunk) )
                                  loc_bin[BIN_NUM-1]++;  // dense bin
   else if (nz_per_row <= 11)     loc_bin[0]++;          // pwarp 
   else if (nz_per_row <= 22)     loc_bin[1]++;          // pwarp
   else if (nz_per_row <= 44)     loc_bin[2]++;          // pwarp
   else if (nz_per_row <= 90)     loc_bin[3]++;          // warp
   else if (nz_per_row <= 180)    loc_bin[4]++;          // tb
   else if (nz_per_row <= 360)    loc_bin[5]++;          // tb
   else if (nz_per_row <= 720)    loc_bin[6]++;          // tb
   else if (nz_per_row <= 1536)   loc_bin[7]++;          // tb
   else if (nz_per_row <= 3584)   loc_bin[8]++;          // tb
   // else if (nz_per_row <= 8192)   loc_bin[9]++;          // tb
   else                           loc_bin[BIN_NUM-2]++;  // dynamic

   #pragma unroll
   for(int i=0;i<BIN_NUM;i++){
      atomicAdd(bin_size+i, loc_bin[i]);
   }
}


//////////////////////////////////////////////////////////////////////////////////////////

__global__ void nsp_set_row_perm_min(int *bin_size, int *bin_offset,
                                 int *max_row_nz, int *row_perm,
                                 int nrows_C, int ncols_C) {

   // retrieve row index
   int i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i >= nrows_C) return;

   // other registers
   int nz_per_row = max_row_nz[i];
   int dest;

   // BINNUM = 11
   float density = (float)nz_per_row / ncols_C * 100.;
   if ( (ncols_C <= MAX_SH_DENSE && density > MIN_DENSITY) || (density > MIN_DENSITY_chunk) ){     // dense bin
      dest = atomicAdd(bin_size +  BIN_NUM-1, 1);
      row_perm[bin_offset[BIN_NUM - 1] + dest] = i;
   }
   else if (nz_per_row <= 11){                                                                     // pwarp 
      dest = atomicAdd(bin_size, 1);
      row_perm[bin_offset[0] + dest] = i;
   }
   else if (nz_per_row <= 22){                                                                     // pwarp 
      dest = atomicAdd(bin_size + 1, 1);
      row_perm[bin_offset[1] + dest] = i;
   }
   else if (nz_per_row <= 44){                                                                     // pwarp 
      dest = atomicAdd(bin_size + 2, 1);
      row_perm[bin_offset[2] + dest] = i;
   }
   else if (nz_per_row <= 90){                                                                     // warp 
      dest = atomicAdd(bin_size + 3, 1);
      row_perm[bin_offset[3] + dest] = i;
   }
   else if (nz_per_row <= 180){                                                                    // tb 
      dest = atomicAdd(bin_size + 4, 1);
      row_perm[bin_offset[4] + dest] = i;
   }
   else if (nz_per_row <= 360){                                                                    // tb
      dest = atomicAdd(bin_size + 5, 1);
      row_perm[bin_offset[5] + dest] = i;
   }
   else if (nz_per_row <= 720){                                                                    // tb
      dest = atomicAdd(bin_size + 6, 1);
      row_perm[bin_offset[6] + dest] = i;
   }
   else if (nz_per_row <= 1536){                                                                   // tb
      dest = atomicAdd(bin_size + 7, 1);
      row_perm[bin_offset[7] + dest] = i;
   }
   else if (nz_per_row <= 3584){                                                                   // tb
      dest = atomicAdd(bin_size + 8, 1);
      row_perm[bin_offset[8] + dest] = i;
   }
   // else if (nz_per_row <= 8192){                                                                   // tb
   //    dest = myatomicAdd(bin_size + 9, 1);
   //    row_perm[bin_offset[9] + dest] = i;
   // }
   else{                                                                                           // dynamic
      dest = atomicAdd(bin_size + BIN_NUM-2, 1);
      row_perm[bin_offset[BIN_NUM-2] + dest] = i;
   }

}

//////////////////////////////////////////////////////////////////////////////////////////

void nsp_set_min_bin( sfBIN *bin, int nrows_C, int ncols_C, int &DIRECT, int &ifrac, float &frac) {

   // set handles
   int *h_bin_offset = bin->h_bin_offset;
   int *h_bin_size   = bin->h_bin_size;
   int *d_row_nz     = bin->d_row_nz;
   int *d_bin_offset = bin->d_bin_offset;
   int *d_bin_size   = bin->d_bin_size;
   int *d_row_perm   = bin->d_row_perm;

   // initialize sfBIN structure to 0
   for (int i = 0; i < BIN_NUM; i++) {
      h_bin_size[i]   = 0;
      h_bin_offset[i] = 0;
   }
   cudaMemset(d_bin_size, 0, BIN_NUM * sizeof(int));

   // Compute size of C rows
   int BS = BLKSIZE_MxM;
   int GS = div_round_up(nrows_C,BS);
   nsp_set_bin_min<<<GS,BS>>>(d_row_nz,d_bin_size,nrows_C,ncols_C);

   // copy group sizes from Device to Host
   cudaMemcpy(h_bin_size, d_bin_size, sizeof(int) * BIN_NUM, cudaMemcpyDeviceToHost);

   // if the largest bin is dominant (has > 15% of the rows) then don't permute the rows and use direct access
   int i = BIN_NUM - 1;
   while (h_bin_size[i] == 0) i--;
   frac  = (float)h_bin_size[i]/nrows_C;
   ifrac = i;
   if ((float)h_bin_size[i]/nrows_C > 0.15 ) { // add condition that it is not the chunk bins
      // nulify the use of other bins
      for (int j = 0; j < i; j++) h_bin_size[j] = 0;
      h_bin_size[i] = nrows_C;      // set up the largest bin to process all the rows
      d_row_perm = nullptr;         // nulify the row permutation pointer
      DIRECT = 1;
   }else{
      DIRECT = 0;
      // reset to 0 group sizes on the Device (recomputed later in set_row_perm)
      cudaMemset(d_bin_size, 0, BIN_NUM * sizeof(int));
      // set-up host
      for (int i = 0; i < BIN_NUM - 1; i++) {
         h_bin_offset[i+1] = h_bin_offset[i] + int(h_bin_size[i]);
      }
      cudaMemcpy(d_bin_offset, h_bin_offset, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);
      nsp_set_row_perm_min<<<GS,BS>>>(d_bin_size,d_bin_offset,d_row_nz,d_row_perm,nrows_C,ncols_C);
      // sort_permutations(d_row_nz, d_row_perm, nrows_C, BIN_NUM, d_bin_offset);
      // sort_permutations(d_row_nz, d_row_perm, nrows_C, 4, d_bin_offset+3);
   }

   #if defined BENCHMARK
      for (int i = 0; i < BIN_NUM; i++) cout << h_bin_size[i] << " ";
      cout << endl;
      // FILE *fid = fopen("bin.txt","a");
      // for (int i = 0; i < BIN_NUM; i++) fprintf(fid,"%d ",h_bin_size[i]);
      // fprintf(fid,"\n");
      // fclose(fid);
   #endif

}


//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////


template <const int pWARP,const int SH_ROW>
__global__ void nsp_calculate_value_col_bin_mpwarp(const int *iat_A,const int *ja_A,const double *coef_A,
                                                   const int * __restrict__ iat_B, const int * __restrict__ ja_B, const double * __restrict__ coef_B,
                                                   const int *iat_C, int *ja_C, double *coef_C,
                                                   const int *row_perm, int *row_nz,
                                                   const int bin_offset, const int nrows) {

   // retrieve thread infos
   int rid  = (row_perm == nullptr) ? (blockIdx.x * (blockDim.x / pWARP) + threadIdx.x / pWARP) : row_perm[(blockIdx.x * (blockDim.x / pWARP) + threadIdx.x / pWARP) + bin_offset];
   int tid  = threadIdx.x & (pWARP - 1);
   int wid  = threadIdx.x / pWARP;
   int wnum = blockDim.x / pWARP;

   // registers
   int jr,kr;
   int je,ke;
   int nz;
   int jcol_A;
   double cval_A,val;
   int key;
   int offset;
   unsigned int count;

   // block shared memory
   extern __shared__ int sh_mem[];
   int *check = (int*) sh_mem;
   double *value = (double*) (&check[wnum*SH_ROW]);

   // initialize hash table
   check = check + wid * SH_ROW;
   value = value + wid * SH_ROW;

   #pragma unroll
   for (jr = tid; jr < SH_ROW; jr += pWARP) {
      check[jr] = -1;
      value[jr] = 0.;
   }

   if (blockIdx.x * (blockDim.x / pWARP) + threadIdx.x / pWARP >= nrows) {
      return;
   }

   // initialize number of non zeros
   if (tid == 0 ) row_nz[rid] = 0;

   // warp synchronization to ensure initialization
   __syncwarp();

   // loop over A-row coefficients
   for (je = iat_A[rid]; je < iat_A[rid + 1]; je++) {
      // load from global memory without using the cache
      jcol_A = load_glob(ja_A + je);
      cval_A = load_glob(coef_A + je);
      // loop over B-row coefficients
      for (ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke+=pWARP) {
         // load from global memory using the cache
         key = ja_B[ke];
         val = coef_B[ke] * cval_A;
         hashmap_bit(check, value, key, val, SH_ROW-1);
      } // end loop over B-row coefficients
   } // end loop over A-row coefficients

   // warp synchronization
   __syncwarp();
   
   // compact hash table
   #pragma unroll
   for (jr = tid; jr < SH_ROW; jr += pWARP) {
      key = check[jr];
      val = value[jr];
      if (key != -1) {
         kr = atomicAdd(row_nz + rid, 1);
         check[kr] = key;
         value[kr] = val;
      }
   }

   // __syncwarp();

   // get non zero terms
   nz = row_nz[rid];

   // Sorting hash table and store data in global memory
   offset = iat_C[rid];
   for (jr = tid; jr < nz; jr += pWARP) {
      key = check[jr];
      count = 0;
      for (kr = 0; kr < nz; kr++) {
         count += (unsigned int)(check[kr] - key) >> 31;
      }
        ja_C[offset + count] = key;
      coef_C[offset + count] = value[jr];
   }

}

//////////////////////////////////////////////////////////////////////////////////////////

template <int SH_ROW>
__global__ void nsp_calculate_value_col_bin_each_warp(const int *iat_A,const int *ja_A,const double *coef_A,
                                                   const int * __restrict__ iat_B, const int * __restrict__ ja_B,const double * __restrict__ coef_B,
                                                   const int *iat_C, int *ja_C, double *coef_C,
                                                   const int *row_perm, int *row_nz,
                                                   const int bin_offset,const int nrows_tb) {

   // retrieve thread infos
   int rid  = (row_perm == nullptr) ? (blockIdx.x * (blockDim.x / WARPSIZE) + threadIdx.x / WARPSIZE) : row_perm[(blockIdx.x * (blockDim.x / WARPSIZE) + threadIdx.x / WARPSIZE) + bin_offset];
   int tid  = threadIdx.x & (WARPSIZE - 1);
   int wid  = threadIdx.x / WARPSIZE;
   int wnum = blockDim.x / WARPSIZE;

   // registers
   int jr,kr;
   int je,ke;
   int jcol_A;
   double cval_A,val;
   int key;
   int nz = 0;
   int offset;
   unsigned int count;

   // block shared memory
   extern __shared__ int sh_mem[];
   int *check = (int*) sh_mem;
   double *value = (double*) (&check[wnum*SH_ROW]);
   check = check + wid * (SH_ROW);
   value = value + wid * (SH_ROW);

   typedef cub::WarpScan<uint8_t> WarpScanT;
   __shared__ typename WarpScanT::TempStorage temp_storage;

   if (blockIdx.x * (blockDim.x / WARPSIZE) + threadIdx.x / WARPSIZE < nrows_tb){
      // initialize hash table
      #pragma unroll
      for (jr = tid; jr < SH_ROW; jr += WARPSIZE) {
         check[jr] = -1;
         value[jr] = 0.;
      }
      // warp synchronization to ensure initialization
      __syncwarp();

      // loop over A-row coefficients
      for (je = iat_A[rid]; je < iat_A[rid + 1]; je++) {
         // load from global memory without using the cache
         jcol_A = load_glob(ja_A + je);
         cval_A = load_glob(coef_A +je);
         // loop over B-row coefficients
         for (ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke += WARPSIZE) {
            // load from global memory using the cache
            key =   ja_B[ke];
            val = coef_B[ke] * cval_A;
            hashmap_bit(check, value, key, val, SH_ROW-1);
         } // end loop over B-row coefficients
      } // end loop over A-row coefficients

      // Thread-Warp synchronization
      __syncwarp();

      // Compact the hash table using cub
      #pragma unroll
      for (jr = tid; jr < SH_ROW; jr += WARPSIZE) {
         key = check[jr];
         val = value[jr];
         uint8_t index = (key<0) ? 0:1;
         uint8_t warp_aggregate;
         WarpScanT(temp_storage).ExclusiveSum(index, index, warp_aggregate);
         if (key != -1){
            check[nz+index] = key;
            value[nz+index] = val;
         }
         nz+=warp_aggregate;
      }

      // __syncwarp();

      // Sorting hash table and store data in global memory
      offset = iat_C[rid];
      for (jr = tid; jr < nz; jr += WARPSIZE) {
         key = check[jr];
         count = 0;
         for (kr = 0; kr < nz; kr++) {
            count += (unsigned int)(check[kr] - key) >> 31;
         }
           ja_C[offset + int(count)] = key;
         coef_C[offset + int(count)] = value[jr];
      }
   }
}


//////////////////////////////////////////////////////////////////////////////////////////

template <const int SH_ROW, const int BS, const int WNUM>
__global__ void nsp_calculate_value_col_bin_each_tb(const int *iat_A,const int *ja_A,const double *coef_A,
                                                    const int * __restrict__ iat_B,const int * __restrict__ ja_B,const double * __restrict__ coef_B,
                                                    const int *iat_C, int *ja_C, double *coef_C,
                                                    const int *row_perm, int *row_nz, const int bin_offset) {

   // retrieve thread infos
   int rid  = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];
   int tid  = threadIdx.x & (WARPSIZE - 1);
   int wid  = threadIdx.x / WARPSIZE;

   // registers
   int jr,kr;
   int je,ke;
   int jcol_A;
   double cval_A,val;
   int key;
   int nz;
   int offset;
   unsigned int count;

   // block shared memory
   extern __shared__ int sh_mem[];
   int *check = (int*) sh_mem;
   double *value = (double*) (&sh_mem[SH_ROW]);

   // initialize hash table
   #pragma unroll 
   for (jr = threadIdx.x; jr < SH_ROW; jr += BS) {
      check[jr] = -1;
      value[jr] = 0.;
   }

   // initialize number of non zeros
   if (threadIdx.x == 0) row_nz[rid] = 0;
   // block synchronization to ensure initialization
   __syncthreads();

   // loop over A-row coefficients
   for (je = iat_A[rid] + wid; je < iat_A[rid + 1]; je += WNUM) {
      // load from global memory without using the cache
      jcol_A = load_glob(ja_A + je);
      cval_A = load_glob(coef_A + je);
      // loop over B-row coefficients
      for (ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke += WARPSIZE) {
         // load from global memory using the cache
         key =   ja_B[ke];
         val = coef_B[ke] * cval_A;
         hashmap_bit(check, value, key, val, SH_ROW-1);
      } // end loop over B-row coefficients
   } // end loop over A-row coefficients
   // Thread-Block synchronization
   __syncthreads();

   // // Compact the hash table using the first warp
   // if (threadIdx.x < WARPSIZE) {
   //    #pragma unroll
   //    for (jr = tid; jr < SH_ROW; jr += WARPSIZE) {
   //       key = check[jr];
   //       val = value[jr];
   //       if (key != -1) {
   //          kr = myatomicAdd(row_nz + rid, 1);
   //          check[kr] = key;
   //          value[kr] = val;
   //       }
   //    }
   // }

   // Compact the hash table
   #pragma unroll
   for (int pass = 0; pass < SH_ROW / BS; pass++) {
      int off = pass*BS;
      key = check[off+threadIdx.x];
      val = value[off+threadIdx.x];
      nz = (key < 0) ? 0 : 1;
      __syncthreads();
      // Compute prefix sum in each warp
      #pragma unroll
      for (int i = 1; i < WARPSIZE; i <<= 1){
         int locSum = __shfl_up_sync(MASKFULL,nz,i,WARPSIZE);
         nz += (tid >= i) ? locSum : 0;
      }

      // Write the sum of the warp into the shared array
      if (tid == WARPSIZE-1) check[off+wid] = nz;
      __syncthreads();

      // First warp computes the blockSum
      if (wid == 0){
         int warpSum = (tid < WNUM) ? check[off+tid] : 0;
         #pragma unroll
         for (int i = 1; i < WARPSIZE; i <<= 1){
            int locSum = __shfl_up_sync(MASKFULL,warpSum,i,WARPSIZE);
            warpSum += (tid >= i) ? locSum : 0;
         }
         check[off+tid] = warpSum;
      }
      __syncthreads();

      nz += (wid > 0) ? row_nz[rid] + check[off+wid-1] : row_nz[rid];

      __syncthreads();
      if (key != -1) {
         check[nz-1] = key;
         value[nz-1] = val;
      }
      // __syncthreads();

      // store the total running sum
      if (threadIdx.x == BS-1) row_nz[rid] = nz;
      // if (threadIdx.x == BS-1) atomicExch(row_nz+rid,pos);
   }

   __syncthreads();

   // get the number of non-zeros
   nz = row_nz[rid];
   // Sorting hash table and store data in global memory
   offset = iat_C[rid];
   for (jr = threadIdx.x; jr < nz; jr += BS) {
      key = check[jr];
      count = 0;
      for (kr = 0; kr < nz; kr++) {
         count += (unsigned int)(check[kr] - key) >> 31;
      }
        ja_C[offset + int(count)] = key;
      coef_C[offset + int(count)] = value[jr];
   }
}


//////////////////////////////////////////////////////////////////////////////////////////

template <int SH_ROW>
__global__ void nsp_calculate_value_col_bin_each_tb_outsort(const int *iat_A,const int *ja_A,const double *coef_A,
                                                   const int * __restrict__ iat_B, const int * __restrict__ ja_B,const double * __restrict__ coef_B,
                                                   const int *iat_C, int *ja_C, double *coef_C,
                                                   const int *row_perm, int *row_nz,
                                                   const int bin_offset, const int nrows_tb) {

   // retrieve thread infos
   int rid  = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];
   int tid  = threadIdx.x & (WARPSIZE - 1);
   int wid  = threadIdx.x / WARPSIZE;
   int wnum = blockDim.x / WARPSIZE;

   // registers
   int jr,kr;
   int je,ke;
   int jcol_A;
   double cval_A,val;
   int key;
   int nz;
   int offset;

   // block shared memory
   extern __shared__ int sh_mem[];
   int *check = (int*) sh_mem;
   double *value = (double*) (&sh_mem[SH_ROW]);

   // initialize hash table
   #pragma unroll 
   for (jr = threadIdx.x; jr < SH_ROW; jr += blockDim.x) {
       check[jr] = -1;
       value[jr] = 0.;
   }

   // initialize number of non zeros
   if (threadIdx.x == 0) row_nz[rid] = 0;
 
   // block synchronization to ensure initialization
   __syncthreads();

   // loop over A-row coefficients
   for (je = iat_A[rid] + wid; je < iat_A[rid + 1]; je += wnum) {
      // load from global memory without using the cache
      jcol_A = load_glob(ja_A + je);
      cval_A = load_glob(coef_A + je);
      // loop over B-row coefficients
      for (ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke += WARPSIZE) {
         // load from global memory using the cache
         key =   ja_B[ke];
         val = coef_B[ke] * cval_A;
         hashmap_bit(check, value, key, val, SH_ROW-1);
      } // end loop over B-row coefficients
   } // end loop over A-row coefficients

   // Thread-Block synchronization
   __syncthreads();

   // Compatc the hash table using the first warp
   if (threadIdx.x < WARPSIZE) {
      #pragma unroll
      for (jr = tid; jr < SH_ROW; jr += WARPSIZE) {
         key = check[jr];
         val = value[jr];
         if (key != -1) {
            kr = atomicAdd(row_nz + rid, 1);
            check[kr] = key;
            value[kr] = val;
         }
      }
   }

   // get the number of non-zeros
   __syncthreads();
   nz = row_nz[rid];

   // Copy the content of shared memory in global memory for later sorting
   offset = iat_C[rid];
   for (jr = threadIdx.x; jr < nz; jr += blockDim.x){
        ja_C[offset + jr] = check[jr];
      coef_C[offset + jr] = value[jr];
   }

}


//////////////////////////////////////////////////////////////////////////////////////////


#define mymax(a,b) ((a)>(b)?(a):(b))

// this barrier should be at least blockDim.x less than the hash table size in order to not have an if - break statement whithin the while loop
#define barrier_sh_tb 3072

template <int SH_ROW>
__global__ void nsp_calculate_value_col_bin_each_tb_chunk_dynamic(const int *iat_A,const int *ja_A,const double *coef_A,
                                                         const int * __restrict__ iat_B,const int * __restrict__ ja_B,const double * __restrict__ coef_B,
                                                         const int *iat_C, int *ja_C, double *coef_C,
                                                         const int *row_perm, int *row_nz,
                                                         const int bin_offset,const int nrows_tb,
                                                         const int ncols_C){

   /* 
      - this kernel is aimed at sparse and big size rows. The idea is to use a smaller number of chunks to process a row of C. If the row is dense
      then this kernel is not efficient, instead use nsp_calculate_value_col_chunk_B_dense.cuh. The density to use this kernel is set to 30%, however,
      it should be configured more precisely.
      
      - while hashing we count the number of elements added to the hash table and we initialize the column_length (maximal number of B columns traversed). 
         When the hash table becomes overfull - we decrease column_length
      --------------------------------------------------------------------------------------------------------------------------------------------------
      - make compaction with atomic addition to increase the hash table size
   */

   // retrieve thread infos
   int rid = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];
   int tid  = threadIdx.x & (WARPSIZE - 1);
   int wid  = threadIdx.x / WARPSIZE;
   int wnum = blockDim.x / WARPSIZE;

   // registers
   int jr;
   int je,ke;
   int jcol_A,jcol_B;
   double cval_A,cval_B;
   int istrB = 0,iendB;
   int key;
   int offset;
   int nz;
   int column_length;

   // block shared memory
   extern __shared__ int sh_mem[];
   int *check = (int*) sh_mem;
   double *value = (double*) (&sh_mem[SH_ROW]);
   int *sh_sums = (int*) &(value[SH_ROW]);

   // initialize hash tablenz
   #pragma unroll 
   for (jr = threadIdx.x; jr < SH_ROW; jr += blockDim.x) {
       check[jr] = -1;
       value[jr] = 0.;
   }

   // initialize number of non zeros
   if (threadIdx.x == 0 ) {
      // row_nz[rid] = 0;
      sh_sums[32] = 0;
   }

   // block synchronization to ensure check initialization
   __syncthreads();

   // get term offset
   offset = iat_C[rid];

   //                   (                min number of chunks                    )
   column_length  = mymax(ncols_C / ( ((iat_C[rid+1] - offset - 1) / SH_ROW ) + 1 ) / 2 ,SH_ROW) ;

   iendB = column_length;

   while (1) {
      // loop over A-row coefficients
      for (je = iat_A[rid] + wid; je < iat_A[rid + 1]; je += wnum) {
         // load from global memory without using the cache
         jcol_A = load_glob(ja_A + je);
         cval_A = load_glob(coef_A + je);

         // get the offset for a term of the row of A that correponds to the row of B

         // loop over row terms of B
         for (ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke += WARPSIZE) {
            // load from global memory using the cache
            jcol_B = ja_B[ke];
            // check end of the chunk
            if (jcol_B >= iendB ) break;
            if (jcol_B >= istrB){
               cval_B = coef_B[ke];
               key  = jcol_B - istrB;
               hashmap_mod_count(sh_sums+32,check, value, key, cval_A * cval_B, SH_ROW);
            } // end check column index
            if (sh_sums[32] > barrier_sh_tb && column_length > SH_ROW) break;
         } // end loop over row terms of B
         if (sh_sums[32] > barrier_sh_tb && column_length > SH_ROW) break;
      } // end loop over A-row coefficients

      // Thread-Block synchronization
      __syncthreads();
      
      // check if the hash table is not overfull, else rerun with a smaller column_length
      if (sh_sums[32] > barrier_sh_tb && column_length > SH_ROW){
         iendB -= column_length;
         column_length /= 2;
         column_length = mymax(SH_ROW,column_length);
         iendB += column_length;

         goto table_init;
      }
/*   							OLD COMPACTION
       // get the previous number of non-zeros
      nz = row_nz[rid];

      // Thread-Block synchronization
      __syncthreads();

      // Compatc the hash table using the first warp
      if (threadIdx.x < WARPSIZE) {
         for (jr = tid; jr < SH_ROW; jr += WARPSIZE) {
            __syncwarp(MASKFULL);
            int col = check[jr] + istrB;
            double val = value[jr];
            __syncwarp(MASKFULL);
            if (col - istrB != -1) {
               int index = myatomicAdd(row_nz + rid, 1);
               check[index-nz] = col;
               value[index-nz] = val;
            }
         }
      }

      // get the current number of non-zeros
      __syncthreads();
      nz = row_nz[rid] - nz;
*/
      //                NEW COMPACTION
      // compact the hash table and store the number of nonzero into sh_sums
      dev_compactKeyVal_inplace(SH_ROW,sh_sums,check,value,sh_sums,istrB);
      __syncthreads();
      nz = *sh_sums;

      // Sorting for shared data and copy to global memory
      for (jr = threadIdx.x; jr < nz; jr += blockDim.x) {
         key = check[jr];
         int count = 0;
         for (int kr = 0; kr < nz; kr++) {
            count += (unsigned int)(check[kr] - key) >> 31;
         }
           ja_C[offset + count] = key;
         coef_C[offset + count] = value[jr];
      }
      __syncthreads();  
      
      // check end of loop over B chunks
      if ( iendB >= ncols_C ) break;

      // update B indeces
      istrB  = iendB;
      iendB += column_length;

      // update term offset
      offset += nz;

      table_init: ;

      // initialize shared scratch
      #pragma unroll 
      for (jr = threadIdx.x; jr < SH_ROW; jr += blockDim.x) {
         check[jr] = -1;
         value[jr] = 0.;
      }
      if (threadIdx.x == 0) sh_sums[32] = 0;

      // synchronize before next chunk cycle
      __syncthreads();

   } // end loop over B chunks

}

#undef barrier_sh_tb 


//////////////////////////////////////////////////////////////////////////////////////////

static __device__ __forceinline__ void dev_compactVal(const int nn,const unsigned char *check,const double *Key, int *sh_sums, double *coef_C, int *ja_C,const int tid,const int wid, const int wnum){

   // Other variables
   int locSum,warpSum,blockSum;

   // First thread inits the first total running sum to zero
   if (threadIdx.x == 0) sh_sums[warpSize] = 0;

   // Loop over all elements using 1 thread per element
   for (int jj = threadIdx.x; jj < nn; jj += blockDim.x){

      // Record "key_value" and decide wether it will deserve or not a position
      double        key_value =   Key[jj];
      unsigned char bit_check = check[jj];
      int pos = (bit_check == 0x00) ? 0:1;
     
      // Compute prefix sum in each warp
      #pragma unroll
      for (int i = 1; i < warpSize; i *= 2){
          locSum = __shfl_up_sync(MASKFULL,pos, i, warpSize);
          if (tid >= i) pos += locSum;
      }

      // Write the sum of the warp into the sh_sums array
      if (tid == warpSize-1) sh_sums[wid] = pos;
      __syncthreads();

      // First warp computes the blockSum
      if (wid == 0){
         warpSum = (tid < wnum) ? sh_sums[tid]:0;
         #pragma unroll
         for (int i = 1; i < warpSize; i *= 2){
            locSum = __shfl_up_sync(MASKFULL,warpSum,i,warpSize);
            if (tid >= i) warpSum += locSum;
         }
         sh_sums[tid] = warpSum;
      }
      __syncthreads();

      // Store in blockSum the running sum correspondig to this warp
      blockSum = (wid > 0) ? sh_sums[wid-1]:0;
      // Add blockSum and the total running sum to determine the position of each entry
      pos += blockSum + sh_sums[warpSize];

      // Store key_value back in Key
      if (bit_check != 0x00) {
         coef_C[pos-1] = key_value;
           ja_C[pos-1] = jj;
      }
      __syncthreads();
      // Store the new total running sum value
      if (threadIdx.x == blockDim.x-1) sh_sums[warpSize] = pos;

   }

}

//////////////////////////////////////////////////////////////////////////////////////////

void prefixSumExclusive(int *in, int nelems, int initVal) {

   thrust::device_ptr<int> d_thrust_in = thrust::device_pointer_cast(in);

	try{
		thrust::exclusive_scan(d_thrust_in, d_thrust_in + nelems+1, d_thrust_in, 0);

	}catch(std::bad_alloc &e){

		printf("Error prefixSumExclusive\n");        
		// exit(EXIT_FAILURE);
      return;

	}

}

//////////////////////////////////////////////////////////////////////////////////////////

__global__ void nsp_calc_C_dense(const int *iat_A,const int *ja_A, const double *coef_A,
                                 const int * __restrict__ iat_B, const int * __restrict__ ja_B, const double * __restrict__ coef_B,
                                 const int *iat_C, int *ja_C, double *coef_C,
                                 const int *row_perm, int *row_nz,int ncols_C,
                                 const int bin_offset, const int nrows_tb, const int SH_ROW) {

   /*
      Use bitarray "check" to count the added elements into shared table called "value".
      We use bitarray to avoid the possibility of accidental zeros in the "value" from reduction.
   */

   // retrieve thread infos
   int rid = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];
   int tid  = threadIdx.x & (warpSize - 1);
   int wid  = threadIdx.x / warpSize;
   int wnum = blockDim.x / warpSize;

   extern __shared__ int sh_mem[];
   unsigned char *check = (unsigned char*)sh_mem;
   double *value = (double*) &(check[SH_ROW]);

   // initialize shared table
   for (int jr = threadIdx.x; jr < ncols_C; jr += blockDim.x) {
       value[jr] = 0.;
       check[jr] = 0x00;
   }

   // block synchronization to ensure initialization
   __syncthreads();

   // loop over A-row coefficients
   for (int je = iat_A[rid] + wid; je < iat_A[rid + 1]; je += wnum) {
      // load from global memory without using the cache
      int jcol_A = load_glob(ja_A + je);
      double    cval_A = load_glob(coef_A + je);
      // loop over B-row coefficients
      for (int ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke += warpSize) {
         // load from global memory using the cache
         int key =   ja_B[ke];
         double cval_B = coef_B[ke];
         // bitmask "check" and update "value"
         check[key] |= 0x01; 
         atomicAdd(value + key, cval_A * cval_B);
      } // end loop over B-row coefficients
   } // end loop over A-row coefficients

   // Thread-Block synchronization
   __syncthreads();
   // New compaction takes the dense row called "value" and bitarray "check" as input and stores the output into csr coef_C[] and ja_C[] 
   int offset = iat_C[rid];
   dev_compactVal(ncols_C,check, value, (int*)&(value[SH_ROW]), coef_C+offset, ja_C+offset, tid, wid, wnum);
}


//////////////////////////////////////////////////////////////////////////////////////////

template <const int BS,const int SH_ROW,const int WNUM>
__global__ void __launch_bounds__(BS, 2) nsp_calculate_value_col_chunk_B_dense(const int *iat_A,const int *ja_A,const double *coef_A,
                                      const int * __restrict__ iat_B, const int * __restrict__ ja_B,const double * __restrict__ coef_B,
                                      const int *iat_C, int *ja_C, double *coef_C,
                                      const int *row_perm, const int bin_offset,const int ncols_C,
                                      const int *d_A_col_offsets, int *d_A_col_chunks) {

   /*
      since we parse the rows of B in chunks size of the shared table, we can use the bitmask as "check" array
      - at each iteration of loop over B terms we store the last visited index of B added into the hash table.
      - in the next chunk iteraion use the offset to go directly to the non visited elements of B.
      - you can use less memory: ideally, the scratch space needed is the number of blocks per SM times the size of the biggest rows of A
      - be careful to not exceed the GPU global memory

      this kernel is configured to use 30 registers when int == int. If you use const __restrict__ then 32
   */

   // retrieve thread infos
   int rid = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];
   int tid  = threadIdx.x & (WARPSIZE - 1);
   int wid  = threadIdx.x / WARPSIZE; 

   // registers
   extern __shared__ int sh_mem[];
   unsigned char *check = (unsigned char*)sh_mem;
   double *value = (double*) &(check[SH_ROW]);
   int *block_sum = (int*)&(value[SH_ROW]);

   // initialize shared table and bitmask
   #pragma unroll 
   for (int jr = threadIdx.x; jr < SH_ROW; jr += BS) {
      value[jr] = 0.;
      check[jr] = 0x00;
   }

   // block synchronization to ensure check initialization
   __syncthreads();

   // get the offset between different rows of C
   int offset = iat_C[rid];

   // get the offset between different rows of A
   int A_row_offset = d_A_col_offsets[blockIdx.x]; // you can delete it together with B_col_offset if you have a direct access 

   // start loop over B chunks
   int iendB  = SH_ROW;

   while (1) {

      // loop over A-row coefficients
      for (int je = iat_A[rid] + wid; je < iat_A[rid + 1]; je += WNUM) {

         // load from global memory without using the cache
         int jcol_A = load_glob(  ja_A + je);
         double    cval_A = load_glob(coef_A + je);

         // get the offset for a term of the row of A that correponds to the row of B
         int B_col_offset = A_row_offset + je - iat_A[rid];
         int iat_B_ind = iat_B[jcol_A];
         
         // loop over row terms of B
         for (int ke = d_A_col_chunks[B_col_offset] + iat_B_ind + tid; ke < iat_B[jcol_A + 1]; ke += WARPSIZE) {

            // load from global memory using the cache
            int jcol_B = ja_B[ke];

            // check end of the chunk
            if ( jcol_B >= iendB ) break;
            else{
               int key = jcol_B - (iendB-SH_ROW); // if you set to short int -> it will use 2 extra registers

               // add the keys to bitmask and update the values in shared table
               check[key] |= 0x01;
               atomicAdd(value + key, cval_A * coef_B[ke]);
               
               // update the B column offsets
               atomicMax((int *)(d_A_col_chunks + B_col_offset), ke - iat_B_ind + 1);
         
            } // end check column index

         } // end loop over row terms of B

      } // end loop over A-row coefficients

      // thread-block synchronization
      __syncthreads(); 
      
      // compact the shared table
      dev_compactVal_chunk_dense<BS,SH_ROW,WNUM>(check,value,block_sum,(iendB-SH_ROW),coef_C+offset,ja_C+offset,tid,wid);

      __syncthreads();

      // check end of loop over B chunks
      if ( iendB >= ncols_C ) break;
     
      // update the offset by adding the number of nonzeros of the previous chunk
      offset += block_sum[32];

      // update B index
      iendB += SH_ROW;

      // initialize the shared table and the bitmask
      #pragma unroll 
      for (int jr = threadIdx.x; jr < SH_ROW; jr += BS) {
         value[jr] = 0.;
         check[jr] = 0x00;
      }

      // synchronize before the next chunk cycle
      __syncthreads();

   } // end loop over B chunks

}


//////////////////////////////////////////////////////////////////////////////////////////

__global__ void set_row_size(int *iat_A, int *d_row_perm, int nn, int bin_shift, int *d_A_col_offsets){

   int tid = threadIdx.x;  
      
   if (d_row_perm == nullptr){
      while(tid < nn){
         d_A_col_offsets[tid] = iat_A[tid+1] - iat_A[tid]; 
         tid += blockDim.x;
      }
   }
   else{
      while(tid < nn){
         int ind = d_row_perm[bin_shift + tid];
         d_A_col_offsets[tid] = iat_A[ind+1] - iat_A[ind]; 
         tid += blockDim.x;
      }
   }
}


//////////////////////////////////////////////////////////////////////////////////////////

void nsp_calculate_value_col_bin( int *d_iat_A, int *d_ja_A, double *d_coef_A,
                                  int *d_iat_B, int *d_ja_B, double *d_coef_B,
                                  int *d_iat_C, int *d_ja_C, double *d_coef_C,
                                  sfBIN *bin, int nrows_C,int ncols_C,int nterm_C,int DIRECT) {
   // set handles
   int *h_bin_offset   = bin->h_bin_offset;
   int *h_bin_size     = bin->h_bin_size;
   int *d_row_perm     = (DIRECT) ? nullptr : bin->d_row_perm;
   int *d_row_nz       = bin->d_row_nz;

   // define varibles for GPU resources
   int GS,BS,SH;
   size_t shmemsize;

   // loop over groups
   for (int i = BIN_NUM - 1; i >= 0; i--) {
      // check sizes
      if (h_bin_size[i] > 0) {
         // select group kernel
         switch (i) {
            case 0: // <= 16
               #if CC == 86
                  BS = 96;
               #else
                  BS = 64;
               #endif
               SH = BS * 16 / 4;
               shmemsize = SH * ( sizeof(int) + sizeof(double) );
               nsp_calculate_value_col_bin_mpwarp<4,16><<<h_bin_size[i]/(BS/4)+1, BS, shmemsize, bin->stream[i]>>>
                                                (d_iat_A,d_ja_A,d_coef_A,
                                                 d_iat_B,d_ja_B,d_coef_B,
                                                 d_iat_C,d_ja_C,d_coef_C,
                                                 d_row_perm,d_row_nz,
                                                 h_bin_offset[i],h_bin_size[i]);
               break;
            case 1: // <= 32
               #if CC == 86
                  BS = 96;
               #else
                  BS = 64;
               #endif
               SH = BS * 32 / 8;
               shmemsize = SH * ( sizeof(int) + sizeof(double) );
               nsp_calculate_value_col_bin_mpwarp<8,32><<<h_bin_size[i]/(BS/8)+1, BS, shmemsize, bin->stream[i]>>>
                                                (d_iat_A,d_ja_A,d_coef_A,
                                                 d_iat_B,d_ja_B,d_coef_B,
                                                 d_iat_C,d_ja_C,d_coef_C,
                                                 d_row_perm,d_row_nz,
                                                 h_bin_offset[i],h_bin_size[i]);
               break;
            case 2 : // <= 64
               #if CC == 86
                  BS = 96;
               #else
                  BS = 64;
               #endif
               SH = BS * 64 / 16;
               shmemsize = SH * ( sizeof(int) + sizeof(double) );
               nsp_calculate_value_col_bin_mpwarp<16,64><<<h_bin_size[i]/(BS/16)+1, BS, shmemsize, bin->stream[i]>>>
                                                (d_iat_A,d_ja_A,d_coef_A,
                                                 d_iat_B,d_ja_B,d_coef_B,
                                                 d_iat_C,d_ja_C,d_coef_C,
                                                 d_row_perm,d_row_nz,
                                                 h_bin_offset[i],h_bin_size[i]);
               break;

            case 3 : // <= 128   
               #if CC == 86
                  BS = 96;
               #else
                  BS = 64;
               #endif
               SH = 128 * BS / 32;
               shmemsize = SH * ( sizeof(int) + sizeof(double) );
               nsp_calculate_value_col_bin_each_warp<128><<<h_bin_size[i]/(BS/32)+1, BS, shmemsize, bin->stream[i]>>>
                                                  (d_iat_A,d_ja_A,d_coef_A,
                                                   d_iat_B,d_ja_B,d_coef_B,
                                                   d_iat_C,d_ja_C,d_coef_C,
                                                   d_row_perm,d_row_nz,
                                                   h_bin_offset[i],h_bin_size[i]);
               break;

            case 4 : // <= 256
               BS = 64;
               GS = h_bin_size[i];
               SH = 256;
               shmemsize = SH * ( sizeof(int) + sizeof(double) );
               nsp_calculate_value_col_bin_each_tb<256,64,2><<<GS, BS, shmemsize, bin->stream[i]>>>
                                                  (d_iat_A,d_ja_A,d_coef_A,
                                                   d_iat_B,d_ja_B,d_coef_B,
                                                   d_iat_C,d_ja_C,d_coef_C,
                                                   d_row_perm,d_row_nz,h_bin_offset[i]);
               break;

            case 5 : // <= 512
               BS = 128;
               GS = h_bin_size[i];
               SH = bin->SHTB_cmp_max / 8;
               shmemsize = SH * ( sizeof(int) + sizeof(double) );
               nsp_calculate_value_col_bin_each_tb<512,128,4><<<GS, BS, shmemsize, bin->stream[i]>>>
                                                  (d_iat_A,d_ja_A,d_coef_A,
                                                   d_iat_B,d_ja_B,d_coef_B,
                                                   d_iat_C,d_ja_C,d_coef_C,
                                                   d_row_perm,d_row_nz,h_bin_offset[i]);
               break;

            case 6 : // <= 1024
               BS = 256;
               GS = h_bin_size[i];
               SH = bin->SHTB_cmp_max / 4;
               shmemsize = SH * ( sizeof(int) + sizeof(double) );
               nsp_calculate_value_col_bin_each_tb<1024,256,8><<<GS, BS, shmemsize, bin->stream[i]>>>
                                                  (d_iat_A,d_ja_A,d_coef_A,
                                                   d_iat_B,d_ja_B,d_coef_B,
                                                   d_iat_C,d_ja_C,d_coef_C,
                                                   d_row_perm,d_row_nz,h_bin_offset[i]);
               break;

            case 7 : // <= 2048
               BS = 512;
               GS = h_bin_size[i];
               SH = bin->SHTB_cmp_max / 2;
               shmemsize = SH * ( sizeof(int) + sizeof(double) );
               nsp_calculate_value_col_bin_each_tb<2048,512,16><<<GS, BS, shmemsize, bin->stream[i]>>>
                                                  (d_iat_A,d_ja_A,d_coef_A,
                                                   d_iat_B,d_ja_B,d_coef_B,
                                                   d_iat_C,d_ja_C,d_coef_C,
                                                   d_row_perm,d_row_nz,h_bin_offset[i]);
               break;

            case 8 : // <= 4096
               BS = 512;
               GS = h_bin_size[i];
               SH = bin->SHTB_cmp_max;
               shmemsize = SH * ( sizeof(int) + sizeof(double) );
               nsp_calculate_value_col_bin_each_tb_outsort<4096><<<GS, BS, shmemsize, bin->stream[i]>>>
                                                  (d_iat_A,d_ja_A,d_coef_A,
                                                   d_iat_B,d_ja_B,d_coef_B,
                                                   d_iat_C,d_ja_C,d_coef_C,
                                                   d_row_perm,d_row_nz,
                                                   h_bin_offset[i],h_bin_size[i]);

               // sort the arrays on global memory using cub library
               {int nSigBits;
               nSigBits = countBITS(ncols_C);         
               nsp_calc_val_sort_rows<<<GS,512,shmemsize,bin->stream[i]>>>(nSigBits,d_row_perm,h_bin_offset[i],
                                                                          d_iat_C,d_ja_C,d_coef_C);  
               }

               break;
            
            // not fully tested, nsp_calc_val_sort_rows should extend up to 16 items per thread
            // case 9 : // <= 8192
            //    GS = h_bin_size[i];
            //    cudaFuncSetAttribute(nsp_calculate_value_col_bin_each_tb_outsort<8192,int>, 
            //                            cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
            //    nsp_calculate_value_col_bin_each_tb_outsort<8192><<<GS, 1024, 98304, bin->stream[i]>>>
            //                                       (d_iat_A,d_ja_A,d_coef_A,
            //                                        d_iat_B,d_ja_B,d_coef_B,
            //                                        d_iat_C,d_ja_C,d_coef_C,
            //                                        d_row_perm,d_row_nz,
            //                                        h_bin_offset[i],h_bin_size[i]);

            //    // sort the arrays on global memory using cub library
            //    {int nSigBits;
            //    nSigBits = countBITS(ncols_C);    
            //    cudaFuncSetAttribute(nsp_calc_val_sort_rows<int>,cudaFuncAttributeMaxDynamicSharedMemorySize,98304);     
            //    nsp_calc_val_sort_rows<<<GS,512,98304,bin->stream[i]>>>(nSigBits,d_row_perm,h_bin_offset[i],
            //                                                               d_iat_C,d_ja_C,d_coef_C);  
            //    }

            //    break;

            case 9 :
               BS = 512;
               GS = h_bin_size[i];
               SH = 4096-512;         // the hash table size must be a multiple of the blocksize, in addition, new compaction takes extra 33*sizeof(int) entries
               shmemsize = 49152;     // max shared memory size   

               nsp_calculate_value_col_bin_each_tb_chunk_dynamic<3584><<<GS, BS, shmemsize, bin->stream[i]>>>
                                                        (d_iat_A,d_ja_A,d_coef_A,
                                                         d_iat_B,d_ja_B,d_coef_B,
                                                         d_iat_C,d_ja_C,d_coef_C,
                                                         d_row_perm,d_row_nz,
                                                         h_bin_offset[i],h_bin_size[i],
                                                         ncols_C);
               
               break;

            case 10 :
               BS = 512;
               GS = h_bin_size[i];
    
               if (ncols_C <= MAX_SH_DENSE){
                  // unfortunately, we have to use the size of shared table as the multiple of the blocksize. Otherwise, the shared table initialization cannot succeed
                  SH = ((ncols_C - 1) / BS + 1) * BS;
                  shmemsize = SH * (sizeof(double)+sizeof(unsigned char)) + (WARPSIZE + 1)*sizeof(int);
                  
		           nsp_calc_C_dense<<<GS, BS, shmemsize, bin->stream[i]>>>
                                                   (d_iat_A,d_ja_A,d_coef_A,
                                                    d_iat_B,d_ja_B,d_coef_B,
                                                    d_iat_C,d_ja_C,d_coef_C,
                                                    d_row_perm,d_row_nz,ncols_C,
                                                    h_bin_offset[i],h_bin_size[i],SH);
               }
               else{
                  int *d_A_bin_col, *d_A_col_offsets, A_bin_terms;  
                  //cudaError_t cudaError;

                  //cudaError = cudaMalloc((void **)&(d_A_col_offsets), (h_bin_size[i]+1)*sizeof(int));
                  //CheckCudaError("nsp_calculate_value_col_bin","allocating d_A_col_offsets",cudaError);
				  checkCudaErrors(cudaMalloc((void **)&(d_A_col_offsets), (h_bin_size[i]+1)*sizeof(int)));
                  
                  // set d_A_col_offsets - to hold the offsets between the rows of A that fit the chunk bin
                  set_row_size<<<1,1024, 0, bin->stream[i]>>>(d_iat_A, d_row_perm,h_bin_size[i], h_bin_offset[i], d_A_col_offsets);

                  // find the cumulative sum of the entries of d_A_col_offsets using thrust
                  prefixSumExclusive(d_A_col_offsets, h_bin_size[i],0);
      
                  cudaMemcpy( &A_bin_terms, &(d_A_col_offsets[h_bin_size[i]]), sizeof(int), cudaMemcpyDeviceToHost );
               
                  //cudaError = cudaMalloc((void **)&(d_A_bin_col), A_bin_terms*sizeof(int));
                  //CheckCudaError("nsp_calculate_value_col_bin","allocating d_A_bin_col",cudaError);
				  checkCudaErrors(cudaMalloc((void **)&(d_A_bin_col), A_bin_terms*sizeof(int)));
                  
                  // initialize the array of d_A_bin_col offsets to store the end of the previous chunk
                  cudaMemset(d_A_bin_col, 0, A_bin_terms*sizeof(int)); // for atomic MAX, alternatively we could use atomic min
                  #if CC == 86
                     #define bs 768
                     #define sh 5376
                  #else
                     #define bs 1024
                     #define sh 5120
                  #endif

                  shmemsize = sh * (sizeof(double)+sizeof(unsigned char)) + (WARPSIZE + 1)*sizeof(int);
                  
                  nsp_calculate_value_col_chunk_B_dense<bs,sh,bs/WARPSIZE><<<GS, bs, shmemsize, bin->stream[i]>>>
                                                                     (d_iat_A,d_ja_A,d_coef_A,
                                                                      d_iat_B,d_ja_B,d_coef_B,
                                                                      d_iat_C,d_ja_C,d_coef_C,
                                                                      d_row_perm,h_bin_offset[i],ncols_C,
                                                                      d_A_col_offsets,d_A_bin_col);

                  //cudaFree(d_A_bin_col);
                  //cudaFree(d_A_col_offsets);
                  cudaFreeAsync(d_A_bin_col,bin->stream[i]);
                  cudaFreeAsync(d_A_col_offsets,bin->stream[i]);
                  }
               break;

            default :
			   printf("nsp_calculate_value_col_bin -- kernel not implemented yet");
               //throw linsol_error ("nsp_calculate_value_col_bin","kernel not implemented yet");          
               break;

         } // end select case
      // }
      } // end check group size
   } // end loop over groups
   // syncronize device
   cudaDeviceSynchronize();
}



//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////


void nsp_spgemm_kernel_hash(sfCSR *a, sfCSR *b, sfCSR *c){
//void spgemm_csrseg_kernel_hash(sfCSR *a, sfCSR *b, sfCSR *c, csrlocinfo *binfo){
    sfBIN bin;
    int nrows_C = a->M;
    int ncols_C = b->N;
    c->M = nrows_C;
    c->N = ncols_C;
  
    // Initialize bin 
	nsp_init_bin (&bin, nrows_C);

    // Set max bin 
	int DIRECT = 0; 
	nsp_set_max_bin ( a->d_rpt, a->d_col, b->d_rpt, &bin, nrows_C, DIRECT);
  
    checkCudaErrors(cudaMalloc((void **)&(c->d_rpt), sizeof(int) * (nrows_C + 1)));

    // Count nz of C
	c->nnz = 0;
	nsp_set_row_nnz( a->d_rpt, a->d_col, b->d_rpt, b->d_col, c->d_rpt,
	                 &bin, nrows_C, ncols_C, &(c->nnz), DIRECT );
		
  
    checkCudaErrors(cudaMalloc((void **)&(c->d_col), sizeof(int) * c->nnz));
    checkCudaErrors(cudaMalloc((void **)&(c->d_val), sizeof(real) * c->nnz));
    
	// Set bin
    int ifrac;
    float frac;
    // compute the exact size of C rows and update sfBIN
    nsp_set_min_bin(&bin, nrows_C, ncols_C, DIRECT, ifrac, frac);
  
  
    // Calculating value of C 
    nsp_calculate_value_col_bin(a->d_rpt, a->d_col, a->d_val,
							    b->d_rpt, b->d_col, b->d_val,
								c->d_rpt, c->d_col, c->d_val,
								&bin, nrows_C, ncols_C, c->nnz, DIRECT);

	nsp_release_bin(&bin);
}
















