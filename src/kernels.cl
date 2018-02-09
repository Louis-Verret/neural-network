#define BLOCK_DIM_32 32
#define BLOCK_DIM_16 16

/********************************************************************************************************/
/************************************* MATRIX OPERATIONS KERNELS ****************************************/
/********************************************************************************************************/


/*
    This kernel realizes the matrix multiplication operation
    It is optimized using block operations to take into account the local size
*/
__kernel void mmul(const int M, const int K, const int N,
    __global double* d_M, __global double* d_N,
    __global double* d_P)
{
  __local double ds_M[BLOCK_DIM_32][BLOCK_DIM_32];
  __local double ds_N[BLOCK_DIM_32][BLOCK_DIM_32];
  int bx = get_group_id(0); int by = get_group_id(1);
  int tx = get_local_id(0); int ty = get_local_id(1);
  // Position de l'element de P sur lequel on travail
  int Row = bx * BLOCK_DIM_32 + tx;
  int Col = by * BLOCK_DIM_32 + ty;
  double Pvalue = 0;
  // Boucle sur l'ensemble les blocs de M et N necessaire pour
  // calculer un element de
  for (int m = 0; m < K/BLOCK_DIM_32; ++m) {
    // Chargement collaboratif en memoire partagee
    const int tiledRow = BLOCK_DIM_32*m + tx;
    const int tiledCol = BLOCK_DIM_32*m + ty;
    ds_M[ty][tx] = d_M[Col*K + tiledRow];
    ds_N[ty][tx] = d_N[tiledCol*M + Row];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < BLOCK_DIM_32; k++) {
      Pvalue += ds_M[ty][k] * ds_N[k][tx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  d_P[Col*M+Row] = Pvalue;
}


/*
    This kernel realizes the multiplication between a matrix and a vector
    It is optimized using block operations to take into account the local size
*/
__kernel void vmul(const int M, const int K,
    __global double* d_M, __global double* d_N,
    __global double* d_P)
{
    __local double ds_M[BLOCK_DIM_32][BLOCK_DIM_32];
    __local double ds_N[BLOCK_DIM_32];
    int bx = get_group_id(0); int by = get_group_id(1);
    int tx = get_local_id(0); int ty = get_local_id(1);

    int Row = bx * BLOCK_DIM_32 + tx;
    int Col = by * BLOCK_DIM_32 + ty;
    double Pvalue = 0;

    for (int m = 0; m < K/BLOCK_DIM_32; ++m) {
      const int tiledRow = BLOCK_DIM_32*m + tx;
      const int tiledCol = BLOCK_DIM_32*m + ty;
      ds_M[ty][tx] = d_M[Col*K + tiledRow];
      ds_N[ty] = d_N[Col];
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k = 0; k < BLOCK_DIM_32; k++) {
        Pvalue += ds_M[ty][k] * ds_N[k];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    d_P[Col] = Pvalue;
}


/*
    This kernel realizes the matrix addition operation
    It is optimized using block operations to take into account the local size
*/
__kernel void add_vector(const int N, const int M, const int sub_N, const int sub_M,
    __global double* mat, __global double* vec,
    __global double* res)
{
    __local double ds_mat[BLOCK_DIM_32][BLOCK_DIM_32];
    __local double ds_vec[BLOCK_DIM_32][BLOCK_DIM_32];
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < M) && (yIndex < N))
    {
        unsigned int index_in = yIndex * M + xIndex;
        ds_mat[get_local_id(1)][get_local_id(0)] = mat[index_in];
        if (get_local_id(1) < sub_N && get_local_id(0) < sub_M) {
            ds_vec[get_local_id(1)][get_local_id(0)] = vec[yIndex];
        } else {
            ds_vec[get_local_id(1)][get_local_id(0)] = 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * BLOCK_DIM_32 + get_local_id(0);
    yIndex = get_group_id(0) * BLOCK_DIM_32 + get_local_id(1);
    if((xIndex < N) && (yIndex < M)) {
        unsigned int index_out = yIndex * N + xIndex;
        res[index_out] = ds_mat[get_local_id(1)][get_local_id(0)] + ds_vec[get_local_id(1)][get_local_id(0)];
    }
}


/*
    This kernel realizes the matrix transpose operation
    It is optimized using block operations to take into account the local size
*/
__kernel void transpose(int height, int width, __global double* idata, __global double* odata, __global double* block)
{
	unsigned int xIndex = get_global_id(0);
	unsigned int yIndex = get_global_id(1);

	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[get_local_id(1)*(BLOCK_DIM_32+1)+get_local_id(0)] = idata[index_in];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	xIndex = get_group_id(1) * BLOCK_DIM_32 + get_local_id(0);
	yIndex = get_group_id(0) * BLOCK_DIM_32 + get_local_id(1);
	if((xIndex < height) && (yIndex < width))
    {
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[get_local_id(0)*(BLOCK_DIM_32+1)+get_local_id(1)];
    }
}


/*
    This kernel realizes the matrix transpose operation (naive implementation)
*/
__kernel void transpose_naive(int height,  int width, __global double* idata, __global double* odata,  __global double* block)
{
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if (xIndex < width && yIndex < height)
    {
        unsigned int index_in  = xIndex + width * yIndex;
        unsigned int index_out = yIndex + height * xIndex;
        odata[index_out] = idata[index_in];
    }
}


/*
    This kernel returns the sum of elements of a matrix
    It is optimized using block operations to take into account the local size
    This is the only kernel that takes more time than the sequentiel code though
*/
__kernel void sum_elements(int height,  int width, __global double *input, __global double *partialSums)
{
    __local double localSums[BLOCK_DIM_16*BLOCK_DIM_16];
    uint local_id_i = get_local_id(0);
    uint local_id_j = get_local_id(1);
    uint group_size_i = get_local_size(0);
    uint group_size_j = get_local_size(1);
    uint global_id_i = get_global_id(0);
    uint global_id_j = get_global_id(1);

    if (global_id_i < height && global_id_j < width)
        localSums[local_id_i + BLOCK_DIM_16*local_id_j] = input[global_id_i + height * global_id_j];

    for (uint stride = (group_size_i*group_size_j)>> 1; stride>0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id_i + BLOCK_DIM_16*local_id_j < stride) {
            localSums[local_id_i + BLOCK_DIM_16*local_id_j] += localSums[local_id_i + BLOCK_DIM_16*local_id_j + stride];
        }
    }

    if (local_id_i == 0 && local_id_j == 0) {
        uint group_id_i = get_group_id(0);
        uint group_id_j = get_group_id(1);
        partialSums[group_id_j*height/BLOCK_DIM_16 + group_id_i] = localSums[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}


/*
    This kernel fills a matrix with zeros
    It is optimized using block operations to take into account the local size
*/
__kernel void fill_with_zeros(int height,  int width, __global double* data)
{
    __local double block[BLOCK_DIM_32][BLOCK_DIM_32];
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[get_local_id(1)][get_local_id(0)] = data[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * BLOCK_DIM_32 + get_local_id(0);
    yIndex = get_group_id(0) * BLOCK_DIM_32 + get_local_id(1);
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        data[index_out] = 0.0;
    }

}


/*
    This kernel realizes the addition matrix operation
    It is optimized using block operations to take into account the local size
*/
__kernel void add(int height,  int width, __global double* idata1, __global double* idata2, __global double *odata)
{
    __local double ds_M[BLOCK_DIM_32][BLOCK_DIM_32];
    __local double ds_N[BLOCK_DIM_32][BLOCK_DIM_32];
    int bx = get_group_id(0); int by = get_group_id(1);
    int tx = get_local_id(0); int ty = get_local_id(1);
    int Row = bx * BLOCK_DIM_32 + tx;
    int Col = by * BLOCK_DIM_32 + ty;
    const int tiledRow = BLOCK_DIM_32*bx + tx;
    const int tiledCol = BLOCK_DIM_32*by + ty;
    ds_M[ty][tx] = idata1[Col*width + tiledRow];
    ds_N[ty][tx] = idata2[tiledCol*height + Row];
    barrier(CLK_LOCAL_MEM_FENCE);
    double Pvalue = ds_M[ty][tx] + ds_N[ty][tx];
    odata[Col*height+Row] = Pvalue;
}


/*
    This kernel realizes the substraction matrix operation
    It is optimized using block operations to take into account the local size
*/
__kernel void sub(int height,  int width, __global double* idata1, __global double* idata2, __global double *odata)
{
    __local double ds_M[BLOCK_DIM_32][BLOCK_DIM_32];
    __local double ds_N[BLOCK_DIM_32][BLOCK_DIM_32];
    int bx = get_group_id(0); int by = get_group_id(1);
    int tx = get_local_id(0); int ty = get_local_id(1);
    int Row = bx * BLOCK_DIM_32 + tx;
    int Col = by * BLOCK_DIM_32 + ty;
    const int tiledRow = BLOCK_DIM_32*bx + tx;
    const int tiledCol = BLOCK_DIM_32*by + ty;
    ds_M[ty][tx] = idata1[Col*width + tiledRow];
    ds_N[ty][tx] = idata2[tiledCol*height + Row];
    barrier(CLK_LOCAL_MEM_FENCE);
    double Pvalue = ds_M[ty][tx] - ds_N[ty][tx];
    odata[Col*height+Row] = Pvalue;
}


/*
    This kernel realizes the  element-wise multiplication matrix operation
    It is optimized using block operations to take into account the local size
*/
__kernel void mul(int height,  int width, __global double* idata1, __global double* idata2, __global double *odata)
{
    __local double ds_M[BLOCK_DIM_32][BLOCK_DIM_32];
    __local double ds_N[BLOCK_DIM_32][BLOCK_DIM_32];
    int bx = get_group_id(0); int by = get_group_id(1);
    int tx = get_local_id(0); int ty = get_local_id(1);
    int Row = bx * BLOCK_DIM_32 + tx;
    int Col = by * BLOCK_DIM_32 + ty;
    const int tiledRow = BLOCK_DIM_32*bx + tx;
    const int tiledCol = BLOCK_DIM_32*by + ty;
    ds_M[ty][tx] = idata1[Col*width + tiledRow];
    ds_N[ty][tx] = idata2[tiledCol*height + Row];
    barrier(CLK_LOCAL_MEM_FENCE);
    double Pvalue = ds_M[ty][tx] * ds_N[ty][tx];
    odata[Col*height+Row] = Pvalue;
}


/*
    This kernel realizes the  element-wise division matrix operation
    It is optimized using block operations to take into account the local size
*/
__kernel void div(int height,  int width, int sub_height,  int sub_width, __global double* idata1, __global double* idata2, __global double *odata)
{
    __local double ds_M[BLOCK_DIM_32][BLOCK_DIM_32];
    __local double ds_N[BLOCK_DIM_32][BLOCK_DIM_32];
    int bx = get_group_id(0); int by = get_group_id(1);
    int tx = get_local_id(0); int ty = get_local_id(1);
    int Row = bx * BLOCK_DIM_32 + tx;
    int Col = by * BLOCK_DIM_32 + ty;
    const int tiledRow = BLOCK_DIM_32*bx + tx;
    const int tiledCol = BLOCK_DIM_32*by + ty;
    ds_M[ty][tx] = idata1[Col*width + tiledRow];
    ds_N[ty][tx] = idata2[tiledCol*height + Row];
    barrier(CLK_LOCAL_MEM_FENCE);
    double Pvalue = ds_M[ty][tx] / ds_N[ty][tx];
    if (Col < sub_height && Row < sub_width) {
        odata[Col*height+Row] = Pvalue;
    } else {
        odata[Col*height+Row] = 0;
    }
}


/*
    This kernel realizes the element-wise addition of a matrix with a double
    It is optimized using block operations to take into account the local size
*/
__kernel void add_coeff(int height,  int width, int sub_height,  int sub_width, __global double* idata, double coeff, __global double *odata)
{
    __local double ds_M[BLOCK_DIM_16][BLOCK_DIM_16];
    int bx = get_group_id(0); int by = get_group_id(1);
    int tx = get_local_id(0); int ty = get_local_id(1);
    int Row = bx * BLOCK_DIM_16 + tx;
    int Col = by * BLOCK_DIM_16 + ty;
    const int tiledRow = BLOCK_DIM_16*bx + tx;
    const int tiledCol = BLOCK_DIM_16*by + ty;
    ds_M[ty][tx] = idata[Col*width + tiledRow];
    barrier(CLK_LOCAL_MEM_FENCE);
    double Pvalue = ds_M[ty][tx] + coeff;
    if (Col < sub_height && Row < sub_width) {
        odata[Col*height+Row] = Pvalue;
    } else {
        odata[Col*height+Row] = 0;
    }
}


/*
    This kernel realizes the element-wise division of a matrix with a double
    It is optimized using block operations to take into account the local size
*/
__kernel void div_coeff(int height,  int width, int sub_height,  int sub_width, __global double* idata, double coeff, __global double *odata)
{
    __local double ds_M[BLOCK_DIM_16][BLOCK_DIM_16];
    int bx = get_group_id(0); int by = get_group_id(1);
    int tx = get_local_id(0); int ty = get_local_id(1);
    int Row = bx * BLOCK_DIM_16 + tx;
    int Col = by * BLOCK_DIM_16 + ty;
    const int tiledRow = BLOCK_DIM_16*bx + tx;
    const int tiledCol = BLOCK_DIM_16*by + ty;
    ds_M[ty][tx] = idata[Col*width + tiledRow];
    barrier(CLK_LOCAL_MEM_FENCE);
    double Pvalue = ds_M[ty][tx] / coeff;
    if (Col < sub_height && Row < sub_width) {
        odata[Col*height+Row] = Pvalue;
    } else {
        odata[Col*height+Row] = 0;
    }
}


/*
    This kernel realizes the element-wise multiplication of a matrix with a double
    It is optimized using block operations to take into account the local size
*/
__kernel void coeff_mul(int height,  int width, double coeff, __global double* idata,  __global double *odata)
{
    __local double ds_M[BLOCK_DIM_16][BLOCK_DIM_16];
    int bx = get_group_id(0); int by = get_group_id(1);
    int tx = get_local_id(0); int ty = get_local_id(1);
    int Row = bx * BLOCK_DIM_16 + tx;
    int Col = by * BLOCK_DIM_16 + ty;
    const int tiledRow = BLOCK_DIM_16*bx + tx;
    const int tiledCol = BLOCK_DIM_16*by + ty;
    ds_M[ty][tx] = idata[Col*width + tiledRow];
    barrier(CLK_LOCAL_MEM_FENCE);
    double Pvalue = coeff * ds_M[ty][tx];
    odata[Col*height+Row] = Pvalue;
}


/*
    This kernel realizes the element-wise substraction of a matrix with a double
    It is optimized using block operations to take into account the local size
*/
__kernel void coeff_sub(int height,  int width, int sub_height, int sub_width, double coeff, __global double* idata,  __global double *odata)
{
    __local double ds_M[BLOCK_DIM_16][BLOCK_DIM_16];
    int bx = get_group_id(0); int by = get_group_id(1);
    int tx = get_local_id(0); int ty = get_local_id(1);
    int Row = bx * BLOCK_DIM_16 + tx;
    int Col = by * BLOCK_DIM_16 + ty;
    const int tiledRow = BLOCK_DIM_16*bx + tx;
    const int tiledCol = BLOCK_DIM_16*by + ty;
    ds_M[ty][tx] = idata[Col*width + tiledRow];
    barrier(CLK_LOCAL_MEM_FENCE);
    double Pvalue = coeff - ds_M[ty][tx];
    if (Col < sub_height && Row < sub_width) {
        odata[Col*height+Row] = Pvalue;
    } else {
        odata[Col*height+Row] = 0;
    }
}


/*
    This kernel realizes the element-wise square root matrix operation
    It is optimized using block operations to take into account the local size
*/
__kernel void sqrt_k(int height,  int width, __global double* idata, __global double *odata)
{
    __local double ds_M[BLOCK_DIM_16][BLOCK_DIM_16];
    int bx = get_group_id(0); int by = get_group_id(1);
    int tx = get_local_id(0); int ty = get_local_id(1);
    int Row = bx * BLOCK_DIM_16 + tx;
    int Col = by * BLOCK_DIM_16 + ty;
    const int tiledRow = BLOCK_DIM_16*bx + tx;
    const int tiledCol = BLOCK_DIM_16*by + ty;
    ds_M[ty][tx] = idata[Col*width + tiledRow];
    barrier(CLK_LOCAL_MEM_FENCE);
    double Pvalue = sqrt(ds_M[ty][tx]);
    odata[Col*height+Row] = Pvalue;
}


/*
    This kernel realizes the element-wise log matrix operation
    It is optimized using block operations to take into account the local size
*/
__kernel void log_k(int height,  int width, int sub_height,  int sub_width, __global double* idata, __global double *odata)
{
    __local double ds_M[BLOCK_DIM_16][BLOCK_DIM_16];
    int bx = get_group_id(0); int by = get_group_id(1);
    int tx = get_local_id(0); int ty = get_local_id(1);
    int Row = bx * BLOCK_DIM_16 + tx;
    int Col = by * BLOCK_DIM_16 + ty;
    const int tiledRow = BLOCK_DIM_16*bx + tx;
    const int tiledCol = BLOCK_DIM_16*by + ty;
    ds_M[ty][tx] = idata[Col*width + tiledRow];
    barrier(CLK_LOCAL_MEM_FENCE);
    double Pvalue = log(ds_M[ty][tx]);
    if (Col < sub_height && Row < sub_width) {
        odata[Col*height+Row] = Pvalue;
    } else {
        odata[Col*height+Row] = 0;
    }
}



/********************************************************************************************************/
/************************************* VECTOR OPERATIONS KERNELS ****************************************/
/********************************************************************************************************/


/*
    This kernel realizes the addition vector operation
    It is optimized using block operations to take into account the local size
*/
__kernel void vector_add(int vector_length, __global double* idata1, __global double* idata2, __global double *odata)
{
    __local double ds_M[BLOCK_DIM_32];
    __local double ds_N[BLOCK_DIM_32];
    unsigned int xIndex = get_global_id(0);

    if (xIndex < vector_length) {
        ds_M[get_local_id(0)] = idata1[xIndex];
        ds_N[get_local_id(0)] = idata2[xIndex];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(0) * BLOCK_DIM_32 + get_local_id(0);
    if (xIndex < vector_length) {
        odata[xIndex] = ds_M[get_local_id(0)] + ds_N[get_local_id(0)];
    }

}


/*
    This kernel realizes the substraction vector operation
    It is optimized using block operations to take into account the local size
*/
__kernel void vector_sub(int vector_length, __global double* idata1, __global double* idata2, __global double *odata)
{
    __local double ds_M[BLOCK_DIM_32];
    __local double ds_N[BLOCK_DIM_32];
    unsigned int xIndex = get_global_id(0);

    if (xIndex < vector_length) {
        ds_M[get_local_id(0)] = idata1[xIndex];
        ds_N[get_local_id(0)] = idata2[xIndex];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(0) * BLOCK_DIM_32 + get_local_id(0);
    if (xIndex < vector_length) {
        odata[xIndex] = ds_M[get_local_id(0)] - ds_N[get_local_id(0)];
    }

}


/*
    This kernel realizes the hadamard product vector operation
    It is optimized using block operations to take into account the local size
*/
__kernel void vector_mul(int vector_length, __global double* idata1, __global double* idata2, __global double *odata)
{
    __local double ds_M[BLOCK_DIM_32];
    __local double ds_N[BLOCK_DIM_32];
    unsigned int xIndex = get_global_id(0);

    if (xIndex < vector_length) {
        ds_M[get_local_id(0)] = idata1[xIndex];
        ds_N[get_local_id(0)] = idata2[xIndex];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(0) * BLOCK_DIM_32 + get_local_id(0);
    if (xIndex < vector_length) {
        odata[xIndex] = ds_M[get_local_id(0)] * ds_N[get_local_id(0)];
    }

}


/*
    This kernel realizes division vector operation
    It is optimized using block operations to take into account the local size
*/
__kernel void vector_div(int vector_length, int sub_vector_length, __global double* idata1, __global double* idata2, __global double *odata)
{
    __local double ds_M[BLOCK_DIM_32];
    __local double ds_N[BLOCK_DIM_32];
    unsigned int xIndex = get_global_id(0);

    if (xIndex < vector_length) {
        ds_M[get_local_id(0)] = idata1[xIndex];
        ds_N[get_local_id(0)] = idata2[xIndex];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(0) * BLOCK_DIM_32 + get_local_id(0);
    if (xIndex < vector_length) {
        if (xIndex < sub_vector_length) {
            odata[xIndex] = ds_M[get_local_id(0)] / ds_N[get_local_id(0)];
        } else {
            odata[xIndex] = 0;
        }
    }

}


/*
    This kernel realizes the element-wise addition of a vector with a double
    It is optimized using block operations to take into account the local size
*/
__kernel void vector_add_coeff(int vector_length, int sub_vector_length, __global double* idata, double coeff, __global double *odata)
{
    __local double block[BLOCK_DIM_32];
    unsigned int xIndex = get_global_id(0);

    if (xIndex < vector_length) {
        block[get_local_id(0)] = idata[xIndex];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(0) * BLOCK_DIM_32 + get_local_id(0);
    if (xIndex < vector_length) {
        if (xIndex < sub_vector_length) {
            odata[xIndex] = block[get_local_id(0)] + coeff;
        } else {
            odata[xIndex] = 0;
        }
    }
}


/*
    This kernel realizes the element-wise division of a vector with a double
    It is optimized using block operations to take into account the local size
*/
__kernel void vector_div_coeff(int vector_length, __global double* idata, double coeff, __global double *odata)
{
    __local double block[BLOCK_DIM_32];
    unsigned int xIndex = get_global_id(0);

    if (xIndex < vector_length) {
        block[get_local_id(0)] = idata[xIndex];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(0) * BLOCK_DIM_32 + get_local_id(0);
    if (xIndex < vector_length) {
        odata[xIndex] = block[get_local_id(0)] / coeff;
    }
}


/*
    This kernel realizes the element-wise coefficient of a vector with a double
    It is optimized using block operations to take into account the local size
*/
__kernel void vector_coeff_mul(int vector_length, double coeff, __global double* idata, __global double *odata)
{
    __local double block[BLOCK_DIM_32];
    unsigned int xIndex = get_global_id(0);

    if (xIndex < vector_length) {
        block[get_local_id(0)] = idata[xIndex];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(0) * BLOCK_DIM_32 + get_local_id(0);
    if (xIndex < vector_length) {
        odata[xIndex] = coeff * block[get_local_id(0)];
    }
}


/*
    This kernel realizes the element-wise square root vector operation
    It is optimized using block operations to take into account the local size
*/
__kernel void vector_sqrt(int vector_length, __global double* idata, __global double *odata)
{
    __local double block[BLOCK_DIM_32];
    unsigned int xIndex = get_global_id(0);

    if (xIndex < vector_length) {
        block[get_local_id(0)] = idata[xIndex];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(0) * BLOCK_DIM_32 + get_local_id(0);
    if (xIndex < vector_length) {
        odata[xIndex] = sqrt(block[get_local_id(0)]);
    }
}


/*
    This kernel fills a vector with zeros
    It is optimized using block operations to take into account the local size
*/
__kernel void vector_fill_with_zeros(int vector_length, __global double* idata)
{
    __local double block[BLOCK_DIM_32];
    unsigned int xIndex = get_global_id(0);

    if (xIndex < vector_length) {
        block[get_local_id(0)] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(0) * BLOCK_DIM_32 + get_local_id(0);
    if (xIndex < vector_length) {
        idata[xIndex] = block[get_local_id(0)];
    }
}


/*
    This kernel fills a matrix with a double
    It is optimized using block operations to take into account the local size
*/
__kernel void vector_fill_with(int vector_length, int sub_vector_length, __global double* idata, double val)
{
    __local double block[BLOCK_DIM_32];
    unsigned int xIndex = get_global_id(0);

    if (xIndex < vector_length) {
        block[get_local_id(0)] = val;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(0) * BLOCK_DIM_32 + get_local_id(0);
    if (xIndex < vector_length) {
        if (xIndex < sub_vector_length) {
            idata[xIndex] = block[get_local_id(0)];
        } else {
            idata[xIndex] = 0;
        }
    }
}



/********************************************************************************************************/
/********************************* ACTIVATION FUNCTION OPERATIONS KERNELS ********************************/
/********************************************************************************************************/


/*
    This kernel realizes the derivative of the linear function element-wise matrix operation
    It is optimized using block operations to take into account the local size
*/
__kernel void linear_dev(int height,  int width, int sub_height,  int sub_width, __global double* idata, __global double *odata)
{
    __local double block[BLOCK_DIM_16][BLOCK_DIM_16];
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[get_local_id(1)][get_local_id(0)] = idata[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * BLOCK_DIM_16 + get_local_id(0);
    yIndex = get_group_id(0) * BLOCK_DIM_16 + get_local_id(1);
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        if (yIndex < sub_height && xIndex < sub_width) {
            odata[index_out] = 1;
        } else {
            odata[index_out] = 0;
        }
    }
}


/*
    This kernel realizes the sigmoid function element-wise matrix operation
    It is optimized using block operations to take into account the local size
*/
__kernel void sigmoid_eval(int height,  int width, int sub_height,  int sub_width, __global double* idata, __global double *odata)
{
    __local double ds_M[BLOCK_DIM_16][BLOCK_DIM_16];
    int bx = get_group_id(0); int by = get_group_id(1);
    int tx = get_local_id(0); int ty = get_local_id(1);
    int Row = bx * BLOCK_DIM_16 + tx;
    int Col = by * BLOCK_DIM_16 + ty;
    const int tiledRow = BLOCK_DIM_16*bx + tx;
    const int tiledCol = BLOCK_DIM_16*by + ty;
    ds_M[ty][tx] = idata[Col*width + tiledRow];
    barrier(CLK_LOCAL_MEM_FENCE);
    double Pvalue = 1.0 / (1.0 + exp(-ds_M[ty][tx]));
    if (Col < sub_height && Row < sub_width) {
        odata[Col*height+Row] = Pvalue;
    } else {
        odata[Col*height+Row] = 0;
    }
}


/*
    This kernel realizes the derivative of the sigmoid function element-wise matrix operation
    It is optimized using block operations to take into account the local size
*/
__kernel void sigmoid_dev(int height,  int width, int sub_height,  int sub_width, __global double* idata, __global double *odata)
{
    __local double ds_M[BLOCK_DIM_16][BLOCK_DIM_16];
    int bx = get_group_id(0); int by = get_group_id(1);
    int tx = get_local_id(0); int ty = get_local_id(1);
    int Row = bx * BLOCK_DIM_16 + tx;
    int Col = by * BLOCK_DIM_16 + ty;
    const int tiledRow = BLOCK_DIM_16*bx + tx;
    const int tiledCol = BLOCK_DIM_16*by + ty;
    ds_M[ty][tx] = idata[Col*width + tiledRow];
    barrier(CLK_LOCAL_MEM_FENCE);
    double eval = 1.0 / (1.0 + exp(-ds_M[ty][tx]));
    double Pvalue = (1 - eval) * eval;
    if (Col < sub_height && Row < sub_width) {
        odata[Col*height+Row] = Pvalue;
    } else {
        odata[Col*height+Row] = 0;
    }
}


/*
    This kernel realizes the tanh function element-wise matrix operation
    It is optimized using block operations to take into account the local size
*/
__kernel void tanh_eval(int height,  int width, int sub_height,  int sub_width, __global double* idata, __global double *odata)
{
    __local double ds_M[BLOCK_DIM_16][BLOCK_DIM_16];
    int bx = get_group_id(0); int by = get_group_id(1);
    int tx = get_local_id(0); int ty = get_local_id(1);
    int Row = bx * BLOCK_DIM_16 + tx;
    int Col = by * BLOCK_DIM_16 + ty;
    const int tiledRow = BLOCK_DIM_16*bx + tx;
    const int tiledCol = BLOCK_DIM_16*by + ty;
    ds_M[ty][tx] = idata[Col*width + tiledRow];
    barrier(CLK_LOCAL_MEM_FENCE);
    double Pvalue = tanh(ds_M[ty][tx]);
    if (Col < sub_height && Row < sub_width) {
        odata[Col*height+Row] = Pvalue;
    } else {
        odata[Col*height+Row] = 0;
    }
}


/*
    This kernel realizes the derivative of the tanh function element-wise matrix operation
    It is optimized using block operations to take into account the local size
*/
__kernel void tanh_dev(int height,  int width, int sub_height,  int sub_width, __global double* idata, __global double *odata)
{
    __local double ds_M[BLOCK_DIM_16][BLOCK_DIM_16];
    int bx = get_group_id(0); int by = get_group_id(1);
    int tx = get_local_id(0); int ty = get_local_id(1);
    int Row = bx * BLOCK_DIM_16 + tx;
    int Col = by * BLOCK_DIM_16 + ty;
    const int tiledRow = BLOCK_DIM_16*bx + tx;
    const int tiledCol = BLOCK_DIM_16*by + ty;
    ds_M[ty][tx] = idata[Col*width + tiledRow];
    barrier(CLK_LOCAL_MEM_FENCE);
    double eval =  tanh(ds_M[ty][tx]);
    double Pvalue = 1 - pow(eval, 2);
    if (Col < sub_height && Row < sub_width) {
        odata[Col*height+Row] = Pvalue;
    } else {
        odata[Col*height+Row] = 0;
    }
}


/*
    This kernel realizes the ReLU (Rectifier Linear Unit) function element-wise matrix operation
    It is optimized using block operations to take into account the local size
*/
__kernel void relu_eval(int height,  int width, int sub_height,  int sub_width, __global double* idata, __global double *odata)
{
    __local double ds_M[BLOCK_DIM_16][BLOCK_DIM_16];
    int bx = get_group_id(0); int by = get_group_id(1);
    int tx = get_local_id(0); int ty = get_local_id(1);
    int Row = bx * BLOCK_DIM_16 + tx;
    int Col = by * BLOCK_DIM_16 + ty;
    const int tiledRow = BLOCK_DIM_16*bx + tx;
    const int tiledCol = BLOCK_DIM_16*by + ty;
    ds_M[ty][tx] = idata[Col*width + tiledRow];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (Col < sub_height && Row < sub_width) {
        double eval = ds_M[ty][tx];
        if (eval > 0) {
            odata[Col*height+Row] = eval;
        } else {
            odata[Col*height+Row] = 0;
        }
    } else {
        odata[Col*height+Row] = 0;
    }
}


/*
    This kernel realizes the derivative of the ReLU function element-wise matrix operation
    It is optimized using block operations to take into account the local size
*/
__kernel void relu_dev(int height,  int width, int sub_height,  int sub_width, __global double* idata, __global double *odata)
{
    __local double ds_M[BLOCK_DIM_16][BLOCK_DIM_16];
    int bx = get_group_id(0); int by = get_group_id(1);
    int tx = get_local_id(0); int ty = get_local_id(1);
    int Row = bx * BLOCK_DIM_16 + tx;
    int Col = by * BLOCK_DIM_16 + ty;
    const int tiledRow = BLOCK_DIM_16*bx + tx;
    const int tiledCol = BLOCK_DIM_16*by + ty;
    ds_M[ty][tx] = idata[Col*width + tiledRow];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (Col < sub_height && Row < sub_width) {
        double eval = ds_M[ty][tx];
        if (eval > 0) {
            odata[Col*height+Row] = 1;
        } else {
            odata[Col*height+Row] = 0;
        }
    } else {
        odata[Col*height+Row] = 0;
    }
}


/*
    This kernel realizes the softmax function matrix operation
    It is optimized using block operations to take into account the local size
*/
__kernel void softmax_eval(int height,  int width, int sub_height,  int sub_width, __global double* sum_exp, __global double* idata, __global double *odata)
{
    __local double ds_M[BLOCK_DIM_16][BLOCK_DIM_16];
    __local double ds_N[BLOCK_DIM_16];
    int bx = get_group_id(0); int by = get_group_id(1);
    int tx = get_local_id(0); int ty = get_local_id(1);
    int Row = bx * BLOCK_DIM_16 + tx;
    int Col = by * BLOCK_DIM_16 + ty;
    const int tiledRow = BLOCK_DIM_16*bx + tx;
    const int tiledCol = BLOCK_DIM_16*by + ty;
    ds_M[ty][tx] = exp(idata[Col*width + tiledRow]);
    ds_N[tx] = sum_exp[Row];
    barrier(CLK_LOCAL_MEM_FENCE);
    double Pvalue = ds_M[ty][tx] / ds_N[tx];
    if (Col < sub_height && Row < sub_width) {
        odata[Col*height+Row] = Pvalue;
    } else {
        odata[Col*height+Row] = 0;
    }
}
