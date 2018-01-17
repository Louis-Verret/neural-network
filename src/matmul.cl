#define TILE_WIDTH 32
__kernel void _mmul(
   const int N,
   __global float* A,
   __global float* B,
   __global float* C)
{
   int ty = get_global_id(0);
   int tx = get_global_id(1);

   // value stores the element that is
   // computed by the thread
   float value = 0;
   for (int k = 0; k < N; ++k)
   {
      float elementA = A[ty * N + k];
      float elementB = B[k * N + tx];
      value += elementA * elementB;
   }

   // Write the matrix to device memory each
   // thread writes one element
   C[ty * N + tx] = value;
}


__kernel void mmul(const int Width,
    __global float* d_M, __global float* d_N,
    __global float* d_P)
{
  __local float ds_M[TILE_WIDTH][TILE_WIDTH];
  __local float ds_N[TILE_WIDTH][TILE_WIDTH];
  int bx = get_group_id(0); int by = get_group_id(1);
  int tx = get_local_id(0); int ty = get_local_id(1);
  // Position de l'element de P sur lequel on travaille
  int Col = bx * TILE_WIDTH + tx;
  int Row = by * TILE_WIDTH + ty;
  float Pvalue = 0;
  // Boucle sur l'ensemble les blocs de M et N necessaire pour
  // calculer un element de
  for (int m = 0; m < Width/TILE_WIDTH; ++m) {
    // Chargement collaboratif en memoire partagee
    ds_M[ty][tx] = d_M[Row*Width + m*TILE_WIDTH+tx];
    ds_N[ty][tx] = d_N[(m*TILE_WIDTH+ty)*Width+Col];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < TILE_WIDTH; ++k)
      Pvalue += ds_M[ty][k] * ds_N[k][tx];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  d_P[Row*Width+Col] = Pvalue;
}
