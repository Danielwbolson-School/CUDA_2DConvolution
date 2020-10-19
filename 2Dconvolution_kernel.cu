#ifndef _2DCONVOLUTION_KERNEL_H_
#define _2DCONVOLUTION_KERNEL_H_

#include <stdio.h>
#include "2Dconvolution.h"

const int32_t KERNEL_WIDTH = 5;
const int32_t numHalo = KERNEL_WIDTH / 2;
const int32_t tileWidth = 32;
const int32_t sharedTileWidth = tileWidth + KERNEL_WIDTH - 1;

// Constant tells the GPU to aggressively cache this variable. Cache coherency isn't an issue
// since these values won't ever change.
__constant__ float Kernel[KERNEL_SIZE][KERNEL_SIZE];

__host__ void ConstantInitialization(float* elements, int32_t size)
{
    cudaMemcpyToSymbol(Kernel, elements, size);
}

// Matrix multiplication kernel thread specification
__global__ void ConvolutionKernel(Matrix N, Matrix P)
{
    int32_t blockX = blockIdx.x; int32_t blockY = blockIdx.y;
    int32_t blockDimX = blockDim.x; int32_t blockDimY = blockDim.y;
    int32_t threadX = threadIdx.x; int32_t threadY = threadIdx.y;

    int32_t row = blockY * blockDimX + threadY;
    int32_t column = blockX * blockDimY + threadX;

    __shared__ float N_s[sharedTileWidth][sharedTileWidth];

    // Store in shared memory.

    // Setting up our shared memory is a bit funky in 2D.
    // Our threads only match up with the middle matrix, but we need to store all values
    // that are marked with an X.
    // +--+--+--+--+--+  +--+--+--+--+--+  +--+--+--+--+--+
    // |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
    // +--------------+  +--------------+  +--------------+
    // |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
    // +--------------+  +--------------+  +--------------+
    // |  |  |  |XX|XX|  |XX|XX|XX|XX|XX|  |XX|XX|  |  |  |
    // +--------------+  +--------------+  +--------------+
    // |  |  |  |XX|XX|  |XX|XX|XX|XX|XX|  |XX|XX|  |  |  |
    // +--+--+--------+  +--------------+  +--------+--+--+
    
    // +--+--+--------+  +--------------+  +--------+--+--+
    // |  |  |  |XX|XX|  |XX|XX|XX|XX|XX|  |XX|XX|  |  |  |
    // +--------------+  +--------------+  +--------------+
    // |  |  |  |XX|XX|  |XX|XX|XX|XX|XX|  |XX|XX|  |  |  |
    // +--------------+  +--------------+  +--------------+
    // |  |  |  |XX|XX|  |XX|XX|XX|XX|XX|  |XX|XX|  |  |  |
    // +--------------+  +--------------+  +--------------+
    // |  |  |  |XX|XX|  |XX|XX|XX|XX|XX|  |XX|XX|  |  |  |
    // +--+--+--------+  +--------------+  +--------+--+--+
    
    // +--+--+--------+  +--------------+  +--------+--+--+
    // |  |  |  |XX|XX|  |XX|XX|XX|XX|XX|  |XX|XX|  |  |  |
    // +--------------+  +--------------+  +--------------+
    // |  |  |  |XX|XX|  |XX|XX|XX|XX|XX|  |XX|XX|  |  |  |
    // +--------------+  +--------------+  +--------------+
    // |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
    // +--------------+  +--------------+  +--------------+
    // |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
    // +--+--+--+--+--+  +--+--+--+--+--+  +--+--+--+--+--+
    
    // An if-check for all 8 surrounding grids. Quite a bit of divergence here, however the
    // memory access savings can potentially be huge, dependent on ratio of image size to kernel size.
    
    // Is it possible that our tile is smaller than the half width of our kernel? I am assuming that
    // tile size is enough to cover the kernel with my 3x3 grid operation below.

    // Only update values that change.

    // Up and Left
    int32_t blockThreadRow = (blockY - 1) * blockDimY + threadY;
    int32_t blockThreadCol = (blockX - 1) * blockDimX + threadX;
    int32_t sharedThreadRow = threadY - (blockDimY - numHalo);
    int32_t sharedThreadCol = threadX - (blockDimX - numHalo);
    int32_t rowOffset = 0; 
    int32_t colOffset = 0;

    if (
        (sharedThreadRow >= 0) && 
        (sharedThreadCol >= 0) && 
        (sharedThreadRow + rowOffset < sharedTileWidth) && 
        (sharedThreadCol + colOffset < sharedTileWidth)
    )
    {
        N_s[sharedThreadRow + rowOffset][sharedThreadCol + colOffset] = 
            ((blockThreadRow < 0) || (blockThreadRow >= N.height) || (blockThreadCol < 0) || (blockThreadCol >= N.width)) ?
                0 : N.elements[blockThreadRow * N.width + blockThreadCol];
    }

    // Up
    //blockThreadRow = (blockY - 1) * blockDimY + threadY;
    blockThreadCol = (blockX) * blockDimX + threadX;
    //sharedThreadRow = threadY - (blockDimY - numHalo);
    sharedThreadCol = threadX;
    //rowOffset = 0; 
    colOffset = numHalo;

    if (
        (sharedThreadRow >= 0) && 
        (sharedThreadCol >= 0) && 
        (sharedThreadRow + rowOffset < sharedTileWidth) && 
        (sharedThreadCol + colOffset < sharedTileWidth)
    )
    {
        N_s[sharedThreadRow + rowOffset][sharedThreadCol + colOffset] = 
            ((blockThreadRow < 0) || (blockThreadRow >= N.height) || (blockThreadCol < 0) || (blockThreadCol >= N.width)) ?
                0 : N.elements[blockThreadRow * N.width + blockThreadCol];
    }

    // Up and Right
    //blockThreadRow = (blockY - 1) * blockDimY + threadY;
    blockThreadCol = (blockX + 1) * blockDimX + threadX;
    //sharedThreadRow = threadY - (blockDimY - numHalo);
   // sharedThreadCol = threadX;
    //rowOffset = 0; 
    colOffset = numHalo + blockDimX;

    if (
        (sharedThreadRow >= 0) && 
        (sharedThreadCol < numHalo) && 
        (sharedThreadRow + rowOffset < sharedTileWidth) && 
        (sharedThreadCol + colOffset < sharedTileWidth)
    )
    {
        N_s[sharedThreadRow + rowOffset][sharedThreadCol + colOffset] = 
            ((blockThreadRow < 0) || (blockThreadRow >= N.height) || (blockThreadCol < 0) || (blockThreadCol >= N.width)) ?
                0 : N.elements[blockThreadRow * N.width + blockThreadCol];
    }

    // Left
    blockThreadRow = (blockY) * blockDimY + threadY;
    blockThreadCol = (blockX - 1) * blockDimX + threadX;
    sharedThreadRow = threadY;
    sharedThreadCol = threadX - (blockDimX - numHalo);
    rowOffset = numHalo; 
    colOffset = 0;

    if (
        (sharedThreadRow >= 0) && 
        (sharedThreadCol >= 0) && 
        (sharedThreadRow + rowOffset < sharedTileWidth) && 
        (sharedThreadCol + colOffset < sharedTileWidth)
    )
    {
        N_s[sharedThreadRow + rowOffset][sharedThreadCol + colOffset] = 
            ((blockThreadRow < 0) || (blockThreadRow >= N.height) || (blockThreadCol < 0) || (blockThreadCol >= N.width)) ?
                0 : N.elements[blockThreadRow * N.width + blockThreadCol];
    }

    // Center
    //blockThreadRow = (blockY) * blockDimY + threadY;
    blockThreadCol = (blockX) * blockDimX + threadX;
    //sharedThreadRow = threadY;
    sharedThreadCol = threadX;
    //rowOffset = numHalo; 
    colOffset = numHalo;

    if (
        (sharedThreadRow >= 0) && 
        (sharedThreadCol >= 0) && 
        (sharedThreadRow + rowOffset < sharedTileWidth) && 
        (sharedThreadCol + colOffset < sharedTileWidth)
    )
    {
        N_s[sharedThreadRow + rowOffset][sharedThreadCol + colOffset] = 
            ((blockThreadRow < 0) || (blockThreadRow >= N.height) || (blockThreadCol < 0) || (blockThreadCol >= N.width)) ?
                0 : N.elements[blockThreadRow * N.width + blockThreadCol];
    }

    // Right
    //blockThreadRow = (blockY) * blockDimY + threadY;
    blockThreadCol = (blockX + 1) * blockDimX + threadX;
    //sharedThreadRow = threadY;
    //sharedThreadCol = threadX;
    //rowOffset = numHalo; 
    colOffset = numHalo + blockDimX;

    if (
        (sharedThreadRow >= 0) && 
        (sharedThreadCol < numHalo) && 
        (sharedThreadRow + rowOffset < sharedTileWidth) && 
        (sharedThreadCol + colOffset < sharedTileWidth)
    )
    {
        N_s[sharedThreadRow + rowOffset][sharedThreadCol + colOffset] = 
            ((blockThreadRow < 0) || (blockThreadRow >= N.height) || (blockThreadCol < 0) || (blockThreadCol >= N.width)) ?
                0 : N.elements[blockThreadRow * N.width + blockThreadCol];
    }

    // Down and Left
    blockThreadRow = (blockY + 1) * blockDimY + threadY;
    blockThreadCol = (blockX - 1) * blockDimX + threadX;
    //sharedThreadRow = threadY;
    sharedThreadCol = threadX - (blockDimX - numHalo);
    rowOffset = numHalo + blockDimY; 
    colOffset = 0;

    if (
        (sharedThreadRow < numHalo) && 
        (sharedThreadCol >= 0) && 
        (sharedThreadRow + rowOffset < sharedTileWidth) && 
        (sharedThreadCol + colOffset < sharedTileWidth)
    )
    {
        N_s[sharedThreadRow + rowOffset][sharedThreadCol + colOffset] = 
            ((blockThreadRow < 0) || (blockThreadRow >= N.height) || (blockThreadCol < 0) || (blockThreadCol >= N.width)) ?
                0 : N.elements[blockThreadRow * N.width + blockThreadCol];
    }

    // Down
    //blockThreadRow = (blockY + 1) * blockDimY + threadY;
    blockThreadCol = (blockX) * blockDimX + threadX;
    //sharedThreadRow = threadY;
    sharedThreadCol = threadX;
    //rowOffset = numHalo + blockDimY; 
    colOffset = numHalo;

    if (
        (sharedThreadRow < numHalo) && 
        (sharedThreadCol >= 0) && 
        (sharedThreadRow + rowOffset < sharedTileWidth) && 
        (sharedThreadCol + colOffset < sharedTileWidth)
    )
    {
        N_s[sharedThreadRow + rowOffset][sharedThreadCol + colOffset] = 
            ((blockThreadRow < 0) || (blockThreadRow >= N.height) || (blockThreadCol < 0) || (blockThreadCol >= N.width)) ?
                0 : N.elements[blockThreadRow * N.width + blockThreadCol];
    }

    // Down and Right
    //blockThreadRow = (blockY + 1) * blockDimY + threadY;
    blockThreadCol = (blockX + 1) * blockDimX + threadX;
    //sharedThreadRow = threadY;
    //sharedThreadCol = threadX;
    //rowOffset = numHalo + blockDimY; 
    colOffset = numHalo + blockDimX;

    if (
        (sharedThreadRow < numHalo) && 
        (sharedThreadCol < numHalo) && 
        (sharedThreadRow + rowOffset < sharedTileWidth) && 
        (sharedThreadCol + colOffset < sharedTileWidth)
    )
    {
        N_s[sharedThreadRow + rowOffset][sharedThreadCol + colOffset] = 
            ((blockThreadRow < 0) || (blockThreadRow >= N.height) || (blockThreadCol < 0) || (blockThreadCol >= N.width)) ?
                0 : N.elements[blockThreadRow * N.width + blockThreadCol];
    }
    __syncthreads();
    
    if (row < P.height && column < P.width)
    {
        // Actual convolution math.
        float pValue = 0;
        
        for (int32_t i = 0; i < KERNEL_WIDTH; i++)
        {
            for (int32_t j = 0; j < KERNEL_WIDTH; j++)
            {
                pValue += Kernel[i][j] * N_s[threadY + i][threadX + j];
            }
        }

        P.elements[row * P.width + column] = pValue;

    }
}

#endif // #ifndef _2DCONVOLUTION_KERNEL_H_
