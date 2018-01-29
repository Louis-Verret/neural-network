#define TILE_WIDTH 32
#define BLOCK_DIM 32
#define BLOCK_DIM_2 16

__kernel void mmul_naive(
   const int N, // size(A) = NxK
   const int K,
   const int M, // size(B) = KxM
   __global double* A,
   __global double* B,
   __global double* C)
{
   int ty = get_global_id(0);
   int tx = get_global_id(1);

   // value stores the element that is
   // computed by the thread
   double value = 0;
   for (int k = 0; k < K; ++k)
   {
      value += A[ty * K + k] * B[k * N + tx];
   }

   // Write the matrix to device memory each
   // thread writes one element
   C[ty * N + tx] = value;
}

__kernel void mmul(const int M, const int K, const int N,
    __global double* d_M, __global double* d_N,
    __global double* d_P)
{
  __local double ds_M[TILE_WIDTH][TILE_WIDTH];
  __local double ds_N[TILE_WIDTH][TILE_WIDTH];
  int bx = get_group_id(0); int by = get_group_id(1);
  int tx = get_local_id(0); int ty = get_local_id(1);
  // Position de l'element de P sur lequel on travail
  int Row = bx * TILE_WIDTH + tx;
  int Col = by * TILE_WIDTH + ty;
  double Pvalue = 0;
  // Boucle sur l'ensemble les blocs de M et N necessaire pour
  // calculer un element de
  for (int m = 0; m < K/TILE_WIDTH; ++m) {
    // Chargement collaboratif en memoire partagee
    const int tiledRow = TILE_WIDTH*m + tx;
    const int tiledCol = TILE_WIDTH*m + ty;
    ds_M[ty][tx] = d_M[Col*K + tiledRow];
    ds_N[ty][tx] = d_N[tiledCol*M + Row];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < TILE_WIDTH; k++) {
      Pvalue += ds_M[ty][k] * ds_N[k][tx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  d_P[Col*M+Row] = Pvalue;
}

__kernel void vmul(const int M, const int K,
    __global double* d_M, __global double* d_N,
    __global double* d_P)
{
    __local double ds_M[TILE_WIDTH][TILE_WIDTH];
    __local double ds_N[TILE_WIDTH];
    int bx = get_group_id(0); int by = get_group_id(1);
    int tx = get_local_id(0); int ty = get_local_id(1);
    // Position de l'element de P sur lequel on travail
    int Row = bx * TILE_WIDTH + tx;
    int Col = by * TILE_WIDTH + ty;
    double Pvalue = 0;
    // Boucle sur l'ensemble les blocs de M et N necessaire pour
    // calculer un element de
    for (int m = 0; m < K/TILE_WIDTH; ++m) {
      // Chargement collaboratif en memoire partagee
      const int tiledRow = TILE_WIDTH*m + tx;
      const int tiledCol = TILE_WIDTH*m + ty;
      ds_M[ty][tx] = d_M[Col*K + tiledRow];
      ds_N[ty] = d_N[Col];
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k = 0; k < TILE_WIDTH; k++) {
        Pvalue += ds_M[ty][k] * ds_N[k];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    d_P[Col] = Pvalue;
}

__kernel void add_vector(const int N, const int M,
    __global double* mat, __global double* vec,
    __global double* res)
{
    __local double ds_mat[BLOCK_DIM][BLOCK_DIM];
    __local double ds_vec[BLOCK_DIM];
    unsigned int xIndex = get_global_id(1);
    unsigned int yIndex = get_global_id(0);

    if((xIndex < M) && (yIndex < N))
    {
        unsigned int index_in = xIndex * M + yIndex;
        ds_mat[get_local_id(1)][get_local_id(0)] = mat[index_in];
        ds_vec[get_local_id(1)] = vec[xIndex];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(0) * BLOCK_DIM + get_local_id(1);
    yIndex = get_group_id(1) * BLOCK_DIM + get_local_id(0);
    if((xIndex < N) && (yIndex < M)) {
        unsigned int index_out = xIndex * M + yIndex;
        res[index_out] = ds_mat[get_local_id(1)][get_local_id(0)] + ds_vec[get_local_id(1)];
    }
}

__kernel void transpose(int height, int width, __global double* idata, __global double* odata, __global double* block)
{
	unsigned int xIndex = get_global_id(0);
	unsigned int yIndex = get_global_id(1);

	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[get_local_id(1)*(BLOCK_DIM+1)+get_local_id(0)] = idata[index_in];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	xIndex = get_group_id(1) * BLOCK_DIM + get_local_id(0);
	yIndex = get_group_id(0) * BLOCK_DIM + get_local_id(1);
	if((xIndex < height) && (yIndex < width))
    {
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[get_local_id(0)*(BLOCK_DIM+1)+get_local_id(1)];
    }
}

__kernel void transpose_naive(int height,  int width, __global double* idata, __global double* odata,  __global double* block)
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

__kernel void sum_elements(int height,  int width, __global double *input, __global double *partialSums)
{
    __local double localSums[BLOCK_DIM*BLOCK_DIM];
    uint local_id_i = get_local_id(0);
    uint local_id_j = get_local_id(1);
    uint group_size_i = get_local_size(0);
    uint group_size_j = get_local_size(1);
    uint global_id_i = get_global_id(0);
    uint global_id_j = get_global_id(1);

    // Copy from global to local memory
    localSums[local_id_i + BLOCK_DIM*local_id_j] = input[global_id_i + height * global_id_j];
    //printf("%i %i %i %i %i %i\n", local_id_i, local_id_j, group_size_i, group_size_j, global_id_i, global_id_j);
    //printf("%i \n", group_size_i * group_size_j);

    // Loop for computing localSums : divide WorkGroup into 2 parts
    for (uint stride = (group_size_i*group_size_j)/2; stride>0; stride /=2) {

    // Waiting for each 2x2 addition into given workgroup
        barrier(CLK_LOCAL_MEM_FENCE);

        // Add elements 2 by 2 between local_id and local_id + stride
        if (local_id_i + BLOCK_DIM*local_id_j < stride) {
            localSums[local_id_i + BLOCK_DIM*local_id_j] += localSums[local_id_i + BLOCK_DIM*local_id_j + stride];
        }

        // Write result into partialSums[nWorkGroups]
        if (local_id_i == 0 && local_id_j == 0) {
            uint group_id_i = get_group_id(0);
            uint group_id_j = get_group_id(1);
            //printf("%i %i \n", group_id_i, group_id_j);
            partialSums[group_id_j*height/BLOCK_DIM + group_id_i] = localSums[0];
        }
    }
}

__kernel void add_naive(int height,  int width, __global double* idata1, __global double* idata2, __global double *odata)
{
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if (xIndex < width && yIndex < height)
    {
        double value = idata1[xIndex * width + yIndex] + idata2[xIndex * width + yIndex];
        odata[xIndex * width + yIndex] = value;
    }
}


__kernel void add(int height,  int width, __global double* idata1, __global double* idata2, __global double *odata)
{
    __local double ds_M[TILE_WIDTH][TILE_WIDTH];
    __local double ds_N[TILE_WIDTH][TILE_WIDTH];
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        ds_M[get_local_id(1)][get_local_id(0)] = idata1[index_in];
        ds_N[get_local_id(1)][get_local_id(0)] = idata2[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * TILE_WIDTH + get_local_id(0);
    yIndex = get_group_id(0) * TILE_WIDTH + get_local_id(1);
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = ds_M[get_local_id(1)][get_local_id(0)] + ds_N[get_local_id(1)][get_local_id(0)];
    }

}

__kernel void sub(int height,  int width, __global double* idata1, __global double* idata2, __global double *odata)
{
    __local double ds_M[TILE_WIDTH][TILE_WIDTH];
    __local double ds_N[TILE_WIDTH][TILE_WIDTH];
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        ds_M[get_local_id(1)][get_local_id(0)] = idata1[index_in];
        ds_N[get_local_id(1)][get_local_id(0)] = idata2[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * TILE_WIDTH + get_local_id(0);
    yIndex = get_group_id(0) * TILE_WIDTH + get_local_id(1);
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = ds_M[get_local_id(1)][get_local_id(0)] - ds_N[get_local_id(1)][get_local_id(0)];
    }

}

__kernel void mul(int height,  int width, __global double* idata1, __global double* idata2, __global double *odata)
{
    __local double ds_M[TILE_WIDTH][TILE_WIDTH];
    __local double ds_N[TILE_WIDTH][TILE_WIDTH];
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        ds_M[get_local_id(1)][get_local_id(0)] = idata1[index_in];
        ds_N[get_local_id(1)][get_local_id(0)] = idata2[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * TILE_WIDTH + get_local_id(0);
    yIndex = get_group_id(0) * TILE_WIDTH + get_local_id(1);
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = ds_M[get_local_id(1)][get_local_id(0)] * ds_N[get_local_id(1)][get_local_id(0)];
    }
}

__kernel void div(int height,  int width, __global double* idata1, __global double* idata2, __global double *odata)
{
    __local double ds_M[TILE_WIDTH][TILE_WIDTH];
    __local double ds_N[TILE_WIDTH][TILE_WIDTH];
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        ds_M[get_local_id(1)][get_local_id(0)] = idata1[index_in];
        ds_N[get_local_id(1)][get_local_id(0)] = idata2[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * TILE_WIDTH + get_local_id(0);
    yIndex = get_group_id(0) * TILE_WIDTH + get_local_id(1);
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = ds_M[get_local_id(1)][get_local_id(0)] / ds_N[get_local_id(1)][get_local_id(0)];
    }
}

__kernel void add_coeff(int height,  int width, __global double* idata, double coeff, __global double *odata)
{
    __local double block[BLOCK_DIM_2][BLOCK_DIM_2];
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[get_local_id(1)][get_local_id(0)] = idata[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * BLOCK_DIM_2 + get_local_id(0);
    yIndex = get_group_id(0) * BLOCK_DIM_2 + get_local_id(1);
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = block[get_local_id(1)][get_local_id(0)] + coeff;
    }
}

__kernel void div_coeff(int height,  int width, __global double* idata, double coeff, __global double *odata)
{
    __local double block[BLOCK_DIM_2][BLOCK_DIM_2];
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[get_local_id(1)][get_local_id(0)] = idata[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * BLOCK_DIM_2 + get_local_id(0);
    yIndex = get_group_id(0) * BLOCK_DIM_2 + get_local_id(1);
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = block[get_local_id(1)][get_local_id(0)] / coeff;
    }
}

__kernel void sqrt_k(int height,  int width, __global double* idata, __global double *odata)
{
    __local double block[BLOCK_DIM_2][BLOCK_DIM_2];
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[get_local_id(1)][get_local_id(0)] = idata[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * BLOCK_DIM_2 + get_local_id(0);
    yIndex = get_group_id(0) * BLOCK_DIM_2 + get_local_id(1);
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = sqrt(block[get_local_id(1)][get_local_id(0)]);
    }
}

__kernel void log_k(int height,  int width, __global double* idata, __global double *odata)
{
    __local double block[BLOCK_DIM_2][BLOCK_DIM_2];
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[get_local_id(1)][get_local_id(0)] = idata[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * BLOCK_DIM_2 + get_local_id(0);
    yIndex = get_group_id(0) * BLOCK_DIM_2 + get_local_id(1);
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = log(block[get_local_id(1)][get_local_id(0)]);
    }
}

__kernel void coeff_mul(int height,  int width, double coeff, __global double* idata,  __global double *odata)
{
    __local double block[BLOCK_DIM_2][BLOCK_DIM_2];
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[get_local_id(1)][get_local_id(0)] = idata[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * BLOCK_DIM_2 + get_local_id(0);
    yIndex = get_group_id(0) * BLOCK_DIM_2 + get_local_id(1);
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = coeff * block[get_local_id(1)][get_local_id(0)];
    }
}

__kernel void coeff_sub(int height,  int width, double coeff, __global double* idata,  __global double *odata)
{
    __local double block[BLOCK_DIM_2][BLOCK_DIM_2];
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[get_local_id(1)][get_local_id(0)] = idata[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * BLOCK_DIM_2 + get_local_id(0);
    yIndex = get_group_id(0) * BLOCK_DIM_2 + get_local_id(1);
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = coeff - block[get_local_id(1)][get_local_id(0)];
    }
}

__kernel void vector_add(int vector_length, __global double* idata1, __global double* idata2, __global double *odata)
{
    __local double ds_M[BLOCK_DIM];
    __local double ds_N[BLOCK_DIM];
    unsigned int xIndex = get_global_id(0);

    if (xIndex < vector_length) {
        ds_M[get_local_id(0)] = idata1[xIndex];
        ds_N[get_local_id(0)] = idata2[xIndex];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(0) * BLOCK_DIM + get_local_id(0);
    if (xIndex < vector_length) {
        odata[xIndex] = ds_M[get_local_id(0)] + ds_N[get_local_id(0)];
    }

}

__kernel void vector_sub(int vector_length, __global double* idata1, __global double* idata2, __global double *odata)
{
    __local double ds_M[BLOCK_DIM];
    __local double ds_N[BLOCK_DIM];
    unsigned int xIndex = get_global_id(0);

    if (xIndex < vector_length) {
        ds_M[get_local_id(0)] = idata1[xIndex];
        ds_N[get_local_id(0)] = idata2[xIndex];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(0) * BLOCK_DIM + get_local_id(0);
    if (xIndex < vector_length) {
        odata[xIndex] = ds_M[get_local_id(0)] - ds_N[get_local_id(0)];
    }

}

__kernel void vector_mul(int vector_length, __global double* idata1, __global double* idata2, __global double *odata)
{
    __local double ds_M[BLOCK_DIM];
    __local double ds_N[BLOCK_DIM];
    unsigned int xIndex = get_global_id(0);

    if (xIndex < vector_length) {
        ds_M[get_local_id(0)] = idata1[xIndex];
        ds_N[get_local_id(0)] = idata2[xIndex];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(0) * BLOCK_DIM + get_local_id(0);
    if (xIndex < vector_length) {
        odata[xIndex] = ds_M[get_local_id(0)] * ds_N[get_local_id(0)];
    }

}

__kernel void vector_div(int vector_length, __global double* idata1, __global double* idata2, __global double *odata)
{
    __local double ds_M[BLOCK_DIM];
    __local double ds_N[BLOCK_DIM];
    unsigned int xIndex = get_global_id(0);

    if (xIndex < vector_length) {
        ds_M[get_local_id(0)] = idata1[xIndex];
        ds_N[get_local_id(0)] = idata2[xIndex];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(0) * BLOCK_DIM + get_local_id(0);
    if (xIndex < vector_length) {
        odata[xIndex] = ds_M[get_local_id(0)] / ds_N[get_local_id(0)];
    }

}

__kernel void vector_add_coeff(int vector_length, __global double* idata, double coeff, __global double *odata)
{
    __local double block[BLOCK_DIM];
    unsigned int xIndex = get_global_id(0);

    if (xIndex < vector_length) {
        block[get_local_id(0)] = idata[xIndex];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(0) * BLOCK_DIM + get_local_id(0);
    if (xIndex < vector_length) {
        odata[xIndex] = block[get_local_id(0)] + coeff;
    }
}

__kernel void vector_div_coeff(int vector_length, __global double* idata, double coeff, __global double *odata)
{
    __local double block[BLOCK_DIM];
    unsigned int xIndex = get_global_id(0);

    if (xIndex < vector_length) {
        block[get_local_id(0)] = idata[xIndex];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(0) * BLOCK_DIM + get_local_id(0);
    if (xIndex < vector_length) {
        odata[xIndex] = block[get_local_id(0)] / coeff;
    }
}

__kernel void vector_coeff_mul(int vector_length, double coeff, __global double* idata, __global double *odata)
{
    __local double block[BLOCK_DIM];
    unsigned int xIndex = get_global_id(0);

    if (xIndex < vector_length) {
        block[get_local_id(0)] = idata[xIndex];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(0) * BLOCK_DIM + get_local_id(0);
    if (xIndex < vector_length) {
        odata[xIndex] = coeff * block[get_local_id(0)];
    }
}

__kernel void vector_sqrt(int vector_length, __global double* idata, __global double *odata)
{
    __local double block[BLOCK_DIM];
    unsigned int xIndex = get_global_id(0);

    if (xIndex < vector_length) {
        block[get_local_id(0)] = idata[xIndex];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(0) * BLOCK_DIM + get_local_id(0);
    if (xIndex < vector_length) {
        odata[xIndex] = sqrt(block[get_local_id(0)]);
    }
}

__kernel void vector_fill_with_zeros(int vector_length, __global double* idata)
{
    __local double block[BLOCK_DIM];
    unsigned int xIndex = get_global_id(0);

    if (xIndex < vector_length) {
        block[get_local_id(0)] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(0) * BLOCK_DIM + get_local_id(0);
    if (xIndex < vector_length) {
        idata[xIndex] = block[get_local_id(0)];
    }
}

__kernel void vector_fill_with(int vector_length, __global double* idata, double val)
{
    __local double block[BLOCK_DIM];
    unsigned int xIndex = get_global_id(0);

    if (xIndex < vector_length) {
        block[get_local_id(0)] = val;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(0) * BLOCK_DIM + get_local_id(0);
    if (xIndex < vector_length) {
        idata[xIndex] = block[get_local_id(0)];
    }
}

__kernel void linear_dev(int height,  int width, __global double* idata, __global double *odata)
{
    __local double block[BLOCK_DIM_2][BLOCK_DIM_2];
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[get_local_id(1)][get_local_id(0)] = idata[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * BLOCK_DIM_2 + get_local_id(0);
    yIndex = get_group_id(0) * BLOCK_DIM_2 + get_local_id(1);
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = 1;
    }
}

__kernel void sigmoid_eval(int height,  int width, __global double* idata, __global double *odata)
{
    __local double block[BLOCK_DIM_2][BLOCK_DIM_2];
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[get_local_id(1)][get_local_id(0)] = idata[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * BLOCK_DIM_2 + get_local_id(0);
    yIndex = get_group_id(0) * BLOCK_DIM_2 + get_local_id(1);
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = 1.0 / (1.0 + exp(-block[get_local_id(1)][get_local_id(0)]));
    }
}

__kernel void sigmoid_dev(int height,  int width, __global double* idata, __global double *odata)
{
    __local double block[BLOCK_DIM_2][BLOCK_DIM_2];
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[get_local_id(1)][get_local_id(0)] = idata[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * BLOCK_DIM_2 + get_local_id(0);
    yIndex = get_group_id(0) * BLOCK_DIM_2 + get_local_id(1);
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        double eval = 1.0 / (1.0 + exp(-block[get_local_id(1)][get_local_id(0)]));
        odata[index_out] = eval * (1- eval);
    }
}

__kernel void tanh_eval(int height,  int width, __global double* idata, __global double *odata)
{
    __local double block[BLOCK_DIM_2][BLOCK_DIM_2];
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[get_local_id(1)][get_local_id(0)] = idata[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * BLOCK_DIM_2 + get_local_id(0);
    yIndex = get_group_id(0) * BLOCK_DIM_2 + get_local_id(1);
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = tanh(block[get_local_id(1)][get_local_id(0)]);
    }
}

__kernel void tanh_dev(int height,  int width, __global double* idata, __global double *odata)
{
    __local double block[BLOCK_DIM_2][BLOCK_DIM_2];
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[get_local_id(1)][get_local_id(0)] = idata[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * BLOCK_DIM_2 + get_local_id(0);
    yIndex = get_group_id(0) * BLOCK_DIM_2 + get_local_id(1);
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        double eval =  tanh(block[get_local_id(1)][get_local_id(0)]);
        odata[index_out] = 1 - pow(eval, 2);
    }
}

__kernel void relu_eval(int height,  int width, __global double* idata, __global double *odata)
{
    __local double block[BLOCK_DIM_2][BLOCK_DIM_2];
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[get_local_id(1)][get_local_id(0)] = idata[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * BLOCK_DIM_2 + get_local_id(0);
    yIndex = get_group_id(0) * BLOCK_DIM_2 + get_local_id(1);
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        double eval = block[get_local_id(1)][get_local_id(0)];
        if (eval > 0) {
            odata[index_out] = eval;
        } else {
            odata[index_out] = 0;
        }
    }
}

__kernel void relu_dev(int height,  int width, __global double* idata, __global double *odata)
{
    __local double block[BLOCK_DIM_2][BLOCK_DIM_2];
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[get_local_id(1)][get_local_id(0)] = idata[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * BLOCK_DIM_2 + get_local_id(0);
    yIndex = get_group_id(0) * BLOCK_DIM_2 + get_local_id(1);
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        double eval = block[get_local_id(1)][get_local_id(0)];
        if (eval > 0) {
            odata[index_out] = 1;
        } else {
            odata[index_out] = 0;
        }
    }
}

__kernel void softmax_eval(int height,  int width, __global double* sum_exp, __global double* idata, __global double *odata)
{
    __local double block[BLOCK_DIM_2][BLOCK_DIM_2];
    __local double block2[BLOCK_DIM_2];
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[get_local_id(1)][get_local_id(0)] = idata[index_in];
        block2[get_local_id(0)] = sum_exp[xIndex];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * BLOCK_DIM_2 + get_local_id(0);
    yIndex = get_group_id(0) * BLOCK_DIM_2 + get_local_id(1);
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = exp(block[get_local_id(1)][get_local_id(0)]) / block2[get_local_id(0)];
    }
}
