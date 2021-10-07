// C++11 multithreading example
// Compute the 2d-integral of x*x*y for x=[-50, 50], y=[20, 40] by dividing the integral over several threads
// Author: Himanshu Goel
//#define PARALLEL_STL
//#define MULTITHREADING

#include <atomic>
#include <thread>
#include <cstdio>
#include <stdlib.h>
#include <chrono>

#ifdef MULTITHREADING
const int thread_count = 16;
#endif

const int x_dim = 4096;
const int y_dim = 4096;

class SourceData {
    public:
        double *in_data;
        double *out_data;
        int mesh_x, mesh_y;

        SourceData(int x_dim, int y_dim)
        {
            in_data = new double[x_dim * y_dim];
            out_data = new double[x_dim * y_dim];

            mesh_x = x_dim;
            mesh_y = y_dim;

            //Initialize input data
            double *in_data_iter = in_data;
            for (int y = 0; y < y_dim; y++)
                for (int x = 0; x < x_dim; x++)
                    *(in_data_iter++) = (double)rand() / RAND_MAX;
        }

        ~SourceData(){
            delete[] in_data;
        }
};

class Convolution {
    public:
        Convolution(){}

        void Apply(SourceData &data, const double *kernel, int kernel_side)
        {
#ifdef MULTITHREADING
            std::thread thds[thread_count];
            int thread_id = 0;
            for (int y = 0; y < data.mesh_y; y += data.mesh_y / thread_count)
            {
                int cur_thread_id = thread_id++;

                thds[cur_thread_id] = std::thread([&data, kernel, kernel_side](int y_start, int y_len)
                {
                    double *out_data_ptr = data.out_data + y_start * data.mesh_x;
                    for (int y = y_start; y < y_start + y_len; y++)
                        for (int x = 0; x < data.mesh_x; x++){
#else
            double *out_data_ptr = data.out_data;
            for (int y = 0; y < data.mesh_y; y++)
                for (int x = 0; x < data.mesh_x; x++){
#endif

                    const double *kernel_iter = kernel;
                    double sum = 0.0;
                    double kernel_sum = 0.0;
                    for (int ky = -kernel_side/2; ky <= kernel_side/2; ky++){
                        for (int kx = -kernel_side/2; kx <= kernel_side/2; kx++){

                            int abs_y = y + ky;
                            int abs_x = x + kx;

                            double kernel_term = *(kernel_iter++);

                            if (abs_x < 0) continue;
                            if (abs_y < 0) continue;
                            if (abs_x >= data.mesh_x) continue;
                            if (abs_y >= data.mesh_y) continue;

                            sum += kernel_term * data.in_data[abs_y * data.mesh_x + abs_x];
                            kernel_sum += kernel_term;
                        }
                    }

                    *(out_data_ptr++) = sum / kernel_sum; //Normalize and store the convolved result
                }
#ifdef MULTITHREADING
                }, y, data.mesh_y / thread_count);
            }

            for (int i = 0; i < thread_count; i++)
                thds[i].join();
#endif
        }
};

int main()
{
    SourceData src(x_dim, y_dim);
    Convolution conv;

    const double kernel[] = 
        {1.0 / 16, 1.0 / 8, 1.0 / 16, 
         1.0 / 8,  1.0 / 4, 1.0 / 8, 
         1.0 / 16, 1.0 / 8, 1.0 / 16};

    double avg_time_taken = 0.0;

    int runs = 0;
    for (; runs < 10; runs++){
        auto start_time = std::chrono::high_resolution_clock::now();
        conv.Apply(src, kernel, 3);
        auto stop_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop_time - start_time;
        avg_time_taken += elapsed.count();
    }
    avg_time_taken /= runs;

    printf("Execution took %f ms\r\n", avg_time_taken);

    return 0;
}