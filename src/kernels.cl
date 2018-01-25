#define TILE_WIDTH 32
#define BLOCK_DIM 32

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

__kernel void transpose(int height, int width, __global float* idata, __global float* odata, __global float* block)
{
    // read the matrix tile into shared memory
	unsigned int xIndex = get_global_id(0);
	unsigned int yIndex = get_global_id(1);

	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[get_local_id(1)*(BLOCK_DIM+1)+get_local_id(0)] = idata[index_in];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// write the transposed matrix tile to global memory
	xIndex = get_group_id(1) * BLOCK_DIM + get_local_id(0);
	yIndex = get_group_id(0) * BLOCK_DIM + get_local_id(1);
	if((xIndex < height) && (yIndex < width))
    {
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[get_local_id(0)*(BLOCK_DIM+1)+get_local_id(1)];
    }
}

__kernel void transpose_naive(int height,  int width, __global float* idata, __global float *odata,  __global float* block)
{
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if (xIndex < width && yIndex < height)
    {
        unsigned int index_in  = xIndex + width * yIndex;
        unsigned int index_out = yIndex + height * xIndex;
        //printf("%i %i %i %i\n", index_out, index_in, height, width);
        odata[index_out] = idata[index_in];
    }
}

__kernel void add_naive(int height,  int width, __global float* idata1, __global float* idata2, __global float *odata)
{
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if (xIndex < width && yIndex < height)
    {
        float value = idata1[xIndex * width + yIndex] + idata2[xIndex * width + yIndex];
        odata[xIndex * width + yIndex] = value;
    }
}
