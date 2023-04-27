// Copyright (c) 2022 Himanshu Goel
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <math.h>
#include <chrono>
#include <complex>
//#include <cublas_v2.h>

using namespace std;
using namespace std::chrono;

void MutualIntensityComponentCUDA_Hg(float* pEx0, float* pEz0, float* pMI0, long nxnz, long itStart, long itEnd, long PerX, long iter0, long tot_iter, int PolCom, bool EhOK, bool EvOK);
void MutualIntensityComponentCUDA_RLi(float* mo, float* v1, float* v2, long nxy, double iter, long long tot_iter, int PloCom);

template <int PolCom>
void mutIntens_CPU(float* pEx0, float* pEz0, float* pMI0, long nxnz, long itStart, long PerX, long iter0, long tot_iter, bool EhOK, bool EvOK)
{
    long long PerArg = nxnz << 1;
    long itEnd = nxnz - 1;

    for (long long iter = iter0; iter < tot_iter; iter++)
    {
        float* pEx = pEx0;
        float* pExT = pEx0;
        float* pEz = pEz0;
        float* pEzT = pEz0;
        pEx += ((iter - iter0) * nxnz * 2);
        pEz += ((iter - iter0) * nxnz * 2);
        pExT += ((iter - iter0) * nxnz * 2);
        pEzT += ((iter - iter0) * nxnz * 2);

        double inv_iter_p_1 = 1. / (iter + 1);
        for (long long it = itStart; it <= itEnd; it++) //OC16042021 (to enable partial update of MI/CSD)
            //for(long long it=0; it<=(itEnd-itStart); it++) //OC03032021 (to enable partial update of MI/CSD)
            //for(long long it=0; it<nxnz; it++)
        {
            float* pMI = pMI0 + (it - itStart) * PerArg; //OC16042021
            //float *pMI = pMI0 + it*PerArg;
            for (long long i = 0; i <= it; i++)
            {
                //if(res = MutualIntensityComponent(pEx, pExT, pEz, pEzT, PolCom, iter, pMI)) return res;

                double ExRe = 0., ExIm = 0., EzRe = 0., EzIm = 0.;
                double ExReT = 0., ExImT = 0., EzReT = 0., EzImT = 0.;
                if (EhOK) { ExRe = *pEx; ExIm = *(pEx + 1); ExReT = *pExT; ExImT = *(pExT + 1); }
                if (EvOK) { EzRe = *pEz; EzIm = *(pEz + 1); EzReT = *pEzT; EzImT = *(pEzT + 1); }
                double ReMI = 0., ImMI = 0.;

                switch (PolCom)
                {
                case 0: // Lin. Hor.
                {
                    //a+b = ExRe + ExIm
                    //c+d = ExReT -ExImT
                    //(ac - bd) = ExRe*ExReT + ExIm*ExImT
                    //(ad + bc) = -ExRe*ExImT + ExIm*ExReT
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
                    double ExRe_p_EzRe = ExRe + EzRe, ExIm_p_EzIm = ExIm + EzIm;
                    double ExRe_p_EzReT = ExReT + EzReT, ExIm_p_EzImT = ExImT + EzImT;
                    ReMI = 0.5 * (ExRe_p_EzRe * ExRe_p_EzReT + ExIm_p_EzIm * ExIm_p_EzImT);
                    ImMI = 0.5 * (ExIm_p_EzIm * ExRe_p_EzReT - ExRe_p_EzRe * ExIm_p_EzImT);
                    break;
                }
                case 3: // Linear 135 deg.
                {
                    double ExRe_mi_EzRe = ExRe - EzRe, ExIm_mi_EzIm = ExIm - EzIm;
                    double ExRe_mi_EzReT = ExReT - EzReT, ExIm_mi_EzImT = ExImT - EzImT;
                    ReMI = 0.5 * (ExRe_mi_EzRe * ExRe_mi_EzReT + ExIm_mi_EzIm * ExIm_mi_EzImT);
                    ImMI = 0.5 * (ExIm_mi_EzIm * ExRe_mi_EzReT - ExRe_mi_EzRe * ExIm_mi_EzImT);
                    break;
                }
                case 5: // Circ. Left //OC08092019: corrected to be in compliance with definitions for right-hand frame (x,z,s) and with corresponding definition and calculation of Stokes params
                    //case 4: // Circ. Right
                {
                    double ExRe_mi_EzIm = ExRe - EzIm, ExIm_p_EzRe = ExIm + EzRe;
                    double ExRe_mi_EzImT = ExReT - EzImT, ExIm_p_EzReT = ExImT + EzReT;
                    ReMI = 0.5 * (ExRe_mi_EzIm * ExRe_mi_EzImT + ExIm_p_EzRe * ExIm_p_EzReT);
                    ImMI = 0.5 * (ExIm_p_EzRe * ExRe_mi_EzImT - ExRe_mi_EzIm * ExIm_p_EzReT);
                    break;
                }
                case 4: // Circ. Right //OC08092019: corrected to be in compliance with definitions for right-hand frame (x,z,s) and with corresponding definition and calculation of Stokes params
                    //case 5: // Circ. Left
                {
                    double ExRe_p_EzIm = ExRe + EzIm, ExIm_mi_EzRe = ExIm - EzRe;
                    double ExRe_p_EzImT = ExReT + EzImT, ExIm_mi_EzReT = ExImT - EzReT;
                    ReMI = 0.5 * (ExRe_p_EzIm * ExRe_p_EzImT + ExIm_mi_EzRe * ExIm_mi_EzReT);
                    ImMI = 0.5 * (ExIm_mi_EzRe * ExRe_p_EzImT - ExRe_p_EzIm * ExIm_mi_EzReT);
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
                if (iter == 0)
                {
                    pMI[0] = (float)ReMI;
                    pMI[1] = (float)ImMI;
                }
                else if (iter > 0)
                {
                    //double iter_p_1 = iter + 1; //OC20012020
                    //long long iter_p_1 = iter + 1;
                    pMI[0] = (float)((pMI[0] * iter + ReMI) * inv_iter_p_1); //OC08052021
                    pMI[1] = (float)((pMI[1] * iter + ImMI) * inv_iter_p_1);
                    //pMI[0] = (float)((pMI[0]*iter + ReMI)/iter_p_1);
                    //pMI[1] = (float)((pMI[1]*iter + ImMI)/iter_p_1);
                }
                else
                {
                    pMI[0] += (float)ReMI;
                    pMI[1] += (float)ImMI;
                }

                pEx += PerX; pEz += PerX;
                pMI += 2;
            }

            pEx = pEx0;
            pEz = pEz0;
            pExT += PerX; pEzT += PerX;
        }
    }
}

int main() {
    const int PolCom = 0;
    const int iter0 = 0;
    const int tot_iter = 1;
    const int nx = 100;
    const int nz = 100;
    const int itStart = 0;
    const bool EhOK = true;
    const bool EvOK = true;

    const bool benchmarking = true;
    const int benchmarking_iter = benchmarking ? 1000 : 1;
    const int cpu_benchmarking_iter = benchmarking ? 100 : 1;
    const int float_pres = 1000;

    printf("Initializing data...\n");
    cudaSetDevice(1);

    //Allocate memory and populate with random data
    float* pEx = new float[tot_iter * nx * nz * 2];
    float* pEz = new float[tot_iter * nx * nz * 2];
    float* pE_tmp = new float[tot_iter * nx * nz * 2];
    float* pMI = new float[(nx * nz - itStart) * nx * nz * 2];
    float* pMI_tmp = new float[(nx * nz - itStart) * nx * nz * 2];

    float* pMI_GPU, * pMI_GPU2, * pEx_GPU, * pEz_GPU;
    cudaMalloc((void**)&pMI_GPU, (nx * nz - itStart) * nx * nz * 2 * sizeof(float));
    cudaMalloc((void**)&pMI_GPU2, (nx * nz - itStart) * nx * nz * 2 * sizeof(float));
    cudaMalloc((void**)&pEx_GPU, tot_iter * nx * nz * 2 * sizeof(float));
    cudaMalloc((void**)&pEz_GPU, tot_iter * nx * nz * 2 * sizeof(float));

    srand(0);
    for (int i = 0; i < tot_iter * nx * nz * 2; i++) {
        pEx[i] = ((rand() % (2 * float_pres)) - float_pres) / (float)float_pres;
        pEz[i] = ((rand() % (2 * float_pres)) - float_pres) / (float)float_pres;
    }
    for (int i = 0; i < (nx * nz - itStart) * nx * nz * 2; i++) pMI[i] = 0.f;

    cudaMemcpy(pEx_GPU, pEx, tot_iter * nx * nz * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pEz_GPU, pEz, tot_iter* nx* nz * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pMI_GPU, pMI, (nx * nz - itStart) * nx * nz * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pMI_GPU2, pMI, (nx * nz - itStart) * nx * nz * 2 * sizeof(float), cudaMemcpyHostToDevice);

    //Create events for timing
    cudaEvent_t start_GPU, stop_GPU;
    cudaEventCreate(&start_GPU);
    cudaEventCreate(&stop_GPU);

    //Setup cuBLAS
    //cublasHandle_t handle;
    //cublasCreate(&handle);
    //cuComplex alpha = make_cuFloatComplex(1, 1);
    //float alpha_f = 1.0f;
    //cuComplex beta = make_cuFloatComplex(0, 0);

    //Call the CPU function
    printf("Running CPU version...\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < cpu_benchmarking_iter; i++)
        mutIntens_CPU<PolCom>(pEx, pEz, pMI, nx * nz, itStart, 2, iter0, tot_iter, EhOK, EvOK);
    auto end = std::chrono::high_resolution_clock::now();

    if (benchmarking)
    {
        std::chrono::duration<double> elapsed = end - start;
        printf("CPU version took %f seconds\n", elapsed.count() / cpu_benchmarking_iter);
    }

    //Call the HG version of the GPU function
    printf("Running HG GPU version...\n");

    cudaEventRecord(start_GPU);
    for (int i = 0; i < benchmarking_iter; i++)
    {
        MutualIntensityComponentCUDA_Hg(pEx_GPU, pEz_GPU, pMI_GPU, nx * nz, itStart, nx * nz, 2, iter0, tot_iter, PolCom, EhOK, EvOK);
        //cublasCtpttr(handle, CUBLAS_FILL_MODE_UPPER, nx * nz, (cuComplex*)pMI_GPU, (cuComplex*)pMI_GPU, nx * nz);
    }
    cudaEventRecord(stop_GPU);

    //Wait for the GPU to finish
    cudaDeviceSynchronize();

    if (benchmarking)
    {
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_GPU, stop_GPU);
        printf("HG GPU version took %f seconds\n", milliseconds / 1000 / benchmarking_iter);
    }

    {
        cudaMemcpy(pMI_tmp, pMI_GPU, (nx * nz - itStart) * nx * nz * 2 * sizeof(float), cudaMemcpyDeviceToHost);

        //Compare the results
        bool error_found = false;
        printf("Comparing results...\n");
        for (int i = 0; i < (nx * nz - itStart) * nx * nz * 2; i++) {
            if (fabs(pMI[i] - pMI_tmp[i]) > 0.0001) {
                int it = i / (nx * nz * 2);
                int i_ = (i % (nx * nz * 2)) / 2;
                int it0 = -it + nx * nz - 1;
                int i0 = i_ + (it0 + 1);
                int warpbase_i = (i0 / 32) * 32;
                std::cout << "Error at it = " << it << " i = " << i_ << " it0 = " << it0 << " i0 = " << i0 << " warpbase_i = " << warpbase_i << std::endl;
                std::cout << "CPU: " << pMI[i] << std::endl;
                std::cout << "GPU: " << pMI_tmp[i] << std::endl;
                std::cout << pEx[i_ * 2] << " + i " << pEx[i_ * 2 + 1] << std::endl;
                error_found = true;
                break;
            }
        }

        if (!error_found) {
            printf("No errors found!\n");
        }
        else
            return 0;
    }

    //Call the RLI version of the GPU function
    printf("Running RLI GPU version...\n");

    cudaEventRecord(start_GPU);
    for (int i = 0; i < benchmarking_iter; i++)
        MutualIntensityComponentCUDA_RLi(pMI_GPU2, pEx_GPU, pEz_GPU, nx * nz, iter0, tot_iter, PolCom);
    cudaEventRecord(stop_GPU);

    //Wait for the GPU to finish
    cudaDeviceSynchronize();

    if (benchmarking)
    {
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_GPU, stop_GPU);
        printf("RLI GPU version took %f seconds\n", milliseconds / 1000 / benchmarking_iter);
    }

    {
        cudaMemcpy(pMI_tmp, pMI_GPU2, (nx * nz - itStart) * nx * nz * 2 * sizeof(float), cudaMemcpyDeviceToHost);

        //Compare the results
        bool error_found = false;
        printf("Comparing results...\n");
        for (int i = 0; i < (nx * nz - itStart) * nx * nz * 2; i++) {
            if (fabs(pMI[i] - pMI_tmp[i]) > 0.0001) {
                std::cout << "Error at index " << i << std::endl;
                std::cout << "CPU: " << pMI[i] << std::endl;
                std::cout << "GPU: " << pMI_tmp[i] << std::endl;
                error_found = true;
                break;
            }
        }

        if (!error_found) {
            printf("No errors found!\n");
        }
        else
            return 0;
    }

    return 0;
}