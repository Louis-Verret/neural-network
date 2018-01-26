#define TILE_WIDTH 32

__kernel void mmul1(
   const int N, // size(A) = NxK
   const int K,
   const int M, // size(B) = KxM
   __global float* A,
   __global float* B,
   __global float* C)
{
   int ty = get_global_id(0);
   int tx = get_global_id(1);

   // value stores the element that is
   // computed by the thread
   float value = 0;
   for (int k = 0; k < K; ++k)
   {
      value += A[ty * K + k] * B[k * N + tx];
   }

   // Write the matrix to device memory each
   // thread writes one element
   C[ty * N + tx] = value;
}

__kernel void mmul(const int M, const int K, const int N,
    __global float* d_M, __global float* d_N,
    __global float* d_P)
{
  __local float ds_M[TILE_WIDTH][TILE_WIDTH];
  __local float ds_N[TILE_WIDTH][TILE_WIDTH];
  int bx = get_group_id(0); int by = get_group_id(1);
  int tx = get_local_id(0); int ty = get_local_id(1);
  // Position de l'element de P sur lequel on travail
  int Row = bx * TILE_WIDTH + tx;
  int Col = by * TILE_WIDTH + ty;
  float Pvalue = 0;
  // Boucle sur l'ensemble les blocs de M et N necessaire pour
  // calculer un element de
  for (int m = 0; m < K/TILE_WIDTH; ++m) {
    // Chargement collaboratif en memoire partagee
    const int tiledRow = TILE_WIDTH*m + tx;
    const int tiledCol = TILE_WIDTH*m + ty;
    ds_M[ty][tx] = d_M[tiledCol*M + Row];
    ds_N[ty][tx] = d_N[Col*K + tiledRow];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < TILE_WIDTH; ++k)
      Pvalue += ds_M[k][tx] * ds_N[ty][k];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  d_P[Col*M+Row] = Pvalue;
}
