#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <assert.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;
using cooperative_groups::thread_group;

template <int perBlk>
__global__ void MutualIntensExtract_PackUnPack(float* pMI0, long nxnz, float iter)
{
	int idx = (blockIdx.x * blockDim.x + threadIdx.x); // nxnz * (nxnz / 2) range

	idx = idx / perBlk * perBlk; //Round idx to perBlk

	int tgt_idx = idx + perBlk;
	for (; idx < tgt_idx; idx++)
		if (idx < nxnz * (nxnz / 2))
		{
			pMI0[idx * 2] = pMI0[idx * 2] * iter;
			pMI0[idx * 2 + 1] = pMI0[idx * 2 + 1] * iter;
		}
}
//Dont need interp mode
__global__ void MutualIntensExtract_v2_Kernel(int* pt_coords, float* pE, float* pMI0, long nxnz, long PerX, long iter)
{
	int pt_idx = blockIdx.x;// +threadIdx.y;

	int block_i = pt_coords[2 * pt_idx + 1];
	int block_it = pt_coords[2 * pt_idx];

	__shared__ float rowR[32];
	__shared__ float rowI[32];

	int c_i = block_i + threadIdx.x; //nxnz range
	int c_it = block_it + threadIdx.x; //nxnz range
	if (c_i < nxnz) 
	{
		rowR[threadIdx.x] = pE[c_i * PerX];
		rowI[threadIdx.x] = pE[c_i * PerX + 1];
	}
	else
	{
		rowR[threadIdx.x] = 0.0f;
		rowI[threadIdx.x] = 0.0f;
	}

	float ExReT = 0.0f;
	float ExImT = 0.0f;
	if (c_it < nxnz)
	{
		ExReT = pE[c_it * PerX];
		ExImT = pE[c_it * PerX + 1];
	}
	else
		return;

	__syncwarp();

	float* pMI = pMI0 + c_it * (nxnz << 1) + (block_i << 1);
	for (int x_i = 0; x_i < 32; x_i++)
	{
		if (block_i + x_i <= c_it)
		{
			float ExRe = rowR[x_i];
			float ExIm = rowI[x_i];

			float ReMI = ExRe * ExReT + ExIm * ExImT;
			float ImMI = ExIm * ExReT - ExRe * ExImT;
			
			//if (block_i + x_i == 0 && c_it == 32)
			//	printf("%f %f, %f %f", ExRe, ExIm, ExReT, ExImT);

			if (iter == 0)
			{
				pMI[0] = (float)ReMI;
				pMI[1] = (float)ImMI;
			}
			else if (iter > 0)
			{
				pMI[0] = (pMI[0] * iter + (float)ReMI) / (float)(iter + 1.);
				pMI[1] = (pMI[1] * iter + (float)ImMI) / (float)(iter + 1.);
			}
			else
			{
				pMI[0] += (float)ReMI;
				pMI[1] += (float)ImMI;
			}

			pMI += 2;
		}
		else
			break;
	}
}

//Dont need interp mode
template <int PolCom, bool EhOK, bool EvOK, bool gt1_TotIter, int itPerBlk>
__global__ void MutualIntensExtract_Kernel(float* pEx0, float* pEz0, float* pMI0, long nxnz, long itStart, long itEnd, long PerX, long iter0, long tot_iter)
{
	thread_block tb = this_thread_block();

	//Calculate coordinates as the typical triangular matrix
	int i0 = (blockIdx.x * blockDim.x + threadIdx.x); //<=nxnz range
	int it0_0 = (blockIdx.y * blockDim.y + threadIdx.y); //nxnz/(2*itPerBlk) range
	long iter = iter0;
	if (gt1_TotIter) iter = ((blockIdx.z * blockDim.z + threadIdx.z) + iter0) % tot_iter; //[iter0, tot_iter) range

	if (i0 > nxnz) return;
	if (it0_0 > nxnz / 2) return;

	for (int it0 = it0_0 * itPerBlk; it0 < it0_0 * itPerBlk + itPerBlk; it0++)
	{
		long it = it0;
		long i = i0;
		if (i0 > it0) //If the coordinates are past the triangular bounds, switch to the lower half of the triangle
		{
			it = nxnz - it0 - 1;
			i = i0 - (it0 + 1);
		}

		if (it >= itEnd) {
			return;
		}

		//float* pMI = pMI0 + it0 * (nxnz << 1) + (i0 << 1); //Compact representation coordinates
		float* pMI = pMI0 + (it - itStart) * (nxnz << 1) + (i << 1); //Full representation coordinates
		float* pEx = pEx0 + i * PerX;
		float* pEz = pEz0 + i * PerX;
		float* pExT = pEx0 + (it - itStart) * PerX;
		float* pEzT = pEz0 + (it - itStart) * PerX;

		if (gt1_TotIter)
		{
			pEx += ((iter - iter0) * nxnz * 2);
			pEz += ((iter - iter0) * nxnz * 2);
			pExT += ((iter - iter0) * nxnz * 2);
			pEzT += ((iter - iter0) * nxnz * 2);
		}

		float ExRe = 0., ExIm = 0., EzRe = 0., EzIm = 0.;
		float ExReT = 0., ExImT = 0., EzReT = 0., EzImT = 0.;

		{
			if (EhOK) { ExRe = *pEx; ExIm = *(pEx + 1); ExReT = *pExT; ExImT = *(pExT + 1); }
			if (EvOK) { EzRe = *pEz; EzIm = *(pEz + 1); EzReT = *pEzT; EzImT = *(pEzT + 1); }
		}
		float ReMI = 0., ImMI = 0.;

		switch (PolCom)
		{
		case 0: // Lin. Hor.
		{
			ReMI = ExRe * ExReT + ExIm * ExImT;
			ImMI = ExIm * ExReT - ExRe * ExImT;
			break;
		}
		case 1: // Lin. Vert.
		{
			ReMI = EzRe * EzReT + EzIm * EzImT;
			ImMI = EzIm * EzReT - EzRe * EzImT;
			break;
		}
		case 2: // Linear 45 deg.
		{
			float ExRe_p_EzRe = ExRe + EzRe, ExIm_p_EzIm = ExIm + EzIm;
			float ExRe_p_EzReT = ExReT + EzReT, ExIm_p_EzImT = ExImT + EzImT;
			ReMI = 0.5f * (ExRe_p_EzRe * ExRe_p_EzReT + ExIm_p_EzIm * ExIm_p_EzImT);
			ImMI = 0.5f * (ExIm_p_EzIm * ExRe_p_EzReT - ExRe_p_EzRe * ExIm_p_EzImT);
			break;
		}
		case 3: // Linear 135 deg.
		{
			float ExRe_mi_EzRe = ExRe - EzRe, ExIm_mi_EzIm = ExIm - EzIm;
			float ExRe_mi_EzReT = ExReT - EzReT, ExIm_mi_EzImT = ExImT - EzImT;
			ReMI = 0.5f * (ExRe_mi_EzRe * ExRe_mi_EzReT + ExIm_mi_EzIm * ExIm_mi_EzImT);
			ImMI = 0.5f * (ExIm_mi_EzIm * ExRe_mi_EzReT - ExRe_mi_EzRe * ExIm_mi_EzImT);
			break;
		}
		case 5: // Circ. Left //OC08092019: corrected to be in compliance with definitions for right-hand frame (x,z,s) and with corresponding definition and calculation of Stokes params
			//case 4: // Circ. Right
		{
			float ExRe_mi_EzIm = ExRe - EzIm, ExIm_p_EzRe = ExIm + EzRe;
			float ExRe_mi_EzImT = ExReT - EzImT, ExIm_p_EzReT = ExImT + EzReT;
			ReMI = 0.5f * (ExRe_mi_EzIm * ExRe_mi_EzImT + ExIm_p_EzRe * ExIm_p_EzReT);
			ImMI = 0.5f * (ExIm_p_EzRe * ExRe_mi_EzImT - ExRe_mi_EzIm * ExIm_p_EzReT);
			break;
		}
		case 4: // Circ. Right //OC08092019: corrected to be in compliance with definitions for right-hand frame (x,z,s) and with corresponding definition and calculation of Stokes params
			//case 5: // Circ. Left
		{
			float ExRe_p_EzIm = ExRe + EzIm, ExIm_mi_EzRe = ExIm - EzRe;
			float ExRe_p_EzImT = ExReT + EzImT, ExIm_mi_EzReT = ExImT - EzReT;
			ReMI = 0.5f * (ExRe_p_EzIm * ExRe_p_EzImT + ExIm_mi_EzRe * ExIm_mi_EzReT);
			ImMI = 0.5f * (ExIm_mi_EzRe * ExRe_p_EzImT - ExRe_p_EzIm * ExIm_mi_EzReT);
			break;
		}
		case -1: // s0
		{
			ReMI = ExRe * ExReT + ExIm * ExImT + EzRe * EzReT + EzIm * EzImT;
			ImMI = ExIm * ExReT - ExRe * ExImT + EzIm * EzReT - EzRe * EzImT;
			break;
		}
		case -2: // s1
		{
			ReMI = ExRe * ExReT + ExIm * ExImT - (EzRe * EzReT + EzIm * EzImT);
			ImMI = ExIm * ExReT - ExRe * ExImT - (EzIm * EzReT - EzRe * EzImT);
			break;
		}
		case -3: // s2
		{
			ReMI = ExImT * EzIm + ExIm * EzImT + ExReT * EzRe + ExRe * EzReT;
			ImMI = ExReT * EzIm - ExRe * EzImT - ExImT * EzRe + ExIm * EzReT;
			break;
		}
		case -4: // s3
		{
			ReMI = ExReT * EzIm + ExRe * EzImT - ExImT * EzRe - ExIm * EzReT;
			ImMI = ExIm * EzImT - ExImT * EzIm - ExReT * EzRe + ExRe * EzReT;
			break;
		}
		default: // total mutual intensity, same as s0
		{
			ReMI = ExRe * ExReT + ExIm * ExImT + EzRe * EzReT + EzIm * EzImT;
			ImMI = ExIm * ExReT - ExRe * ExImT + EzIm * EzReT - EzRe * EzImT;
			break;
			//return CAN_NOT_EXTRACT_MUT_INT;
		}
		}

		if (gt1_TotIter)
		{
			//Gather all values updating the same point in the warp and write them in one go
			int mask = __match_any_sync(__activemask(), ((long long)i0) << 32 | (long long)it0);
			int leader = __ffs(mask) - 1;
			for (int offset = 16; offset > 0; offset >>= 1)
			{
				ReMI += __shfl_down_sync(mask, ReMI, offset);
				ImMI += __shfl_down_sync(mask, ImMI, offset);
			}
			if (threadIdx.x % 32 == leader)
			{
				atomicAdd(&pMI[0], (float)ReMI);
				atomicAdd(&pMI[1], (float)ImMI);
			}
		}
		else
		{
			if (iter == 0)
			{
				pMI[0] = (float)ReMI;
				pMI[1] = (float)ImMI;
			}
			else if (iter > 0)
			{
				pMI[0] = (pMI[0] * iter + (float)ReMI) / (float)(iter + 1.);
				pMI[1] = (pMI[1] * iter + (float)ImMI) / (float)(iter + 1.);
			}
			else
			{
				pMI[0] += (float)ReMI;
				pMI[1] += (float)ImMI;
			}
		}
	}
}

template <int PolCom, bool gt1_TotIter>
void MutualIntensExtract_CUDA_Sub(float* pEx0, float* pEz0, float* pMI0, long nxnz, long itStart, long itEnd, long PerX, long iter0, long tot_iter, bool EhOK, bool EvOK)
{
#define KERNEL_V2
#ifdef KERNEL_V2

	dim3 threads = dim3(32, 1, 1);

	long r_nxnz = nxnz / threads.x + ((nxnz % threads.x) > 0);
	long pt_cnt = (r_nxnz * (r_nxnz + 1)) / 2;

	int* pt_coords = new int[pt_cnt * 2];
	int idx = 0;
	for (int i0 = 0; i0 < r_nxnz; i0++)
		for (int j0 = 0; j0 <= i0; j0++)
		{
			pt_coords[idx++] = i0 * 32;
			pt_coords[idx++] = j0 * 32;
		}
	int* pt_coords_cuda;
	cudaMalloc((void**)&pt_coords_cuda, pt_cnt * 2 * sizeof(int));
	cudaMemcpyAsync(pt_coords_cuda, pt_coords, pt_cnt * 2 * sizeof(int), cudaMemcpyHostToDevice);

	dim3 grid = dim3(pt_cnt / threads.y + ((pt_cnt % threads.y) > 0), 1, 1);

	for (int i = iter0; i < tot_iter; i++)
		if (PolCom == 0)
			MutualIntensExtract_v2_Kernel<< <grid, threads >> > (pt_coords_cuda, pEx0, pMI0, nxnz, PerX, i);
		else if (PolCom == 1)
			MutualIntensExtract_v2_Kernel << <grid, threads >> > (pt_coords_cuda, pEz0, pMI0, nxnz, PerX, i);

	cudaFreeAsync(pt_coords_cuda, 0);

#else
	const int itPerBlk = 1;
	dim3 threads = dim3(1, 1, 1);
	if (gt1_TotIter && (tot_iter - iter0) >= 16)
		threads = dim3(48, 1, 16);
	else
		threads = dim3(48, 16, 1);
	dim3 grid = dim3((nxnz + 1) / threads.x + (threads.x > 1), (nxnz / 2) / (threads.y * itPerBlk) + (threads.y > 1), (tot_iter - iter0) / threads.z + (threads.z > 1));

	if (EhOK)
	{
		if (EvOK) MutualIntensExtract_Kernel<PolCom, true, true, gt1_TotIter, itPerBlk> << <grid, threads >> > (pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter);
		else MutualIntensExtract_Kernel<PolCom, true, false, gt1_TotIter, itPerBlk> << <grid, threads >> > (pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter);
	}
	else
	{
		if (EvOK) MutualIntensExtract_Kernel<PolCom, false, true, gt1_TotIter, itPerBlk> << <grid, threads >> > (pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter);
		else MutualIntensExtract_Kernel<PolCom, false, false, gt1_TotIter, itPerBlk> << <grid, threads >> > (pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter);
	}
#endif
}

void MutualIntensityComponentCUDA_Hg(float* pEx0, float* pEz0, float* pMI0, long nxnz, long itStart, long itEnd, long PerX, long iter0, long tot_iter, int PolCom, bool EhOK, bool EvOK)
{
	if (tot_iter - iter0 > 1)
	{
		const int perBlk = 32;
		dim3 threads = dim3(256, 1, 1);
		dim3 grid = dim3((nxnz / (threads.x * perBlk)) + 1, 1, 1);
		if (nxnz % (threads.x * perBlk) == 0) grid.x--;
		//Un-average mutual intensity
		MutualIntensExtract_PackUnPack<perBlk> << <grid, threads >> > (pMI0, nxnz, iter0);

		//Perform mutual intensity extraction
		switch (PolCom)
		{
		case  0: MutualIntensExtract_CUDA_Sub<  0, true>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		case  1: MutualIntensExtract_CUDA_Sub<  1, true>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		case  2: MutualIntensExtract_CUDA_Sub<  2, true>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		case  3: MutualIntensExtract_CUDA_Sub<  3, true>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		case  4: MutualIntensExtract_CUDA_Sub<  4, true>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		case  5: MutualIntensExtract_CUDA_Sub<  5, true>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		case -1: MutualIntensExtract_CUDA_Sub< -1, true>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		case -2: MutualIntensExtract_CUDA_Sub< -2, true>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		case -3: MutualIntensExtract_CUDA_Sub< -3, true>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		case -4: MutualIntensExtract_CUDA_Sub< -4, true>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		default: MutualIntensExtract_CUDA_Sub< -5, true>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		}

		//Re-average mutual intensity
		if (tot_iter > 1) MutualIntensExtract_PackUnPack<perBlk> << <grid, threads >> > (pMI0, nxnz, 1.0 / tot_iter);

	}
	else if (tot_iter - iter0 == 1)
	{
		//No need for separate handling of averaging, the main kernel can handle it
		switch (PolCom)
		{
		case  0: MutualIntensExtract_CUDA_Sub<  0, false>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		case  1: MutualIntensExtract_CUDA_Sub<  1, false>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		case  2: MutualIntensExtract_CUDA_Sub<  2, false>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		case  3: MutualIntensExtract_CUDA_Sub<  3, false>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		case  4: MutualIntensExtract_CUDA_Sub<  4, false>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		case  5: MutualIntensExtract_CUDA_Sub<  5, false>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		case -1: MutualIntensExtract_CUDA_Sub< -1, false>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		case -2: MutualIntensExtract_CUDA_Sub< -2, false>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		case -3: MutualIntensExtract_CUDA_Sub< -3, false>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		case -4: MutualIntensExtract_CUDA_Sub< -4, false>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		default: MutualIntensExtract_CUDA_Sub< -5, false>(pEx0, pEz0, pMI0, nxnz, itStart, itEnd, PerX, iter0, tot_iter, EhOK, EvOK); break;
		}
	}
	else
	{
		printf("Error: MutualIntensExtract_CUDA: tot_iter - iter0 <= 0");
	}
}