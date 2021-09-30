// C++11 multithreading example
// Compute the 2d-integral of x*x*y for x=[-50, 50], y=[20, 40] by dividing the integral over several threads
// Author: Himanshu Goel

#include <atomic>
#include <thread>
#include <mutex>
#include <shared_mutex>

const double x_min = -50;
const double x_max = 50;
const double y_min = 20;
const double y_max = 40;
const int x_steps = 10000;
const int y_steps = 10000;
const double dx = 1.0 / (x_max - x_min);
const double dy = 1.0 / (y_max - y_min);
const double x_stepsz = (x_max - x_min) / x_steps;

#define MULTITHREADED

#ifdef MULTITHREADED
const int thread_count = 8;
#else
const int thread_count = 1;
#endif

std::atomic_uint32_t y_counter_atomic;  //Atomic int - basic math operations (addition/multiplication etc) for integers act as a single unit, preventing thread race conditions
uint32_t y_counter_nonatomic;   //Regular int used to show how atomic operations differ

double integral = 0;    //Stores the result of the integration calculation
std::mutex summation_lock; //synchronization used to prevent a race condition when updating the 'integral' variable

//The function run on separate threads
void thread_func(double y_local_min, double y_local_max)
{
    double sum = 0;
    double y_stepsz = (y_local_max - y_local_min) / ((double)y_steps / thread_count);
    for (double y = y_local_min + y_stepsz * 0.5; y < y_local_max; y += y_stepsz) {

        double x_sum = 0;
        for (double x = x_min + x_stepsz * 0.5; x < x_max; x += x_stepsz)   //Use the midpoint rule to compute the integral of x*x
            x_sum += x * x * x_stepsz;

        sum += x_sum * y * y_stepsz;    //Use the midpoint rule to compute the complete x*x*y integral

        y_counter_nonatomic++;  //This acts as 3 separate operations: read-increment-write
        y_counter_atomic++;     //This acts as a single operation
    }

    summation_lock.lock();      //Code between lock and unlock can only be executed by one thread at a time
    integral += sum;            //Update the integral calculation
    summation_lock.unlock();    //Release the mutex so another thread can obtain the lock
}

int main()
{
    printf("Solving integral  x*x*y for x=[%f, %f] y=[%f, %f]\r\n", x_min, x_max, y_min, y_max);

    y_counter_atomic = 0;
    y_counter_nonatomic = 0;

#ifdef MULTITHREADED
    //Create threads
    printf("Using %d threads.\r\n", thread_count);
    std::thread** threads = new std::thread * [thread_count];

    double y_sz = (y_max - y_min) / thread_count;
    for (int i = 0; i < thread_count; i++) {
        threads[i] = new std::thread(thread_func, y_min + i * y_sz, y_min + (i + 1) * y_sz);    //Call thread_func(y_min + i * y_sz, y_min + (i + 1) * y_sz) on a new thread
    }

    for (int i = 0; i < thread_count; i++)
    {
        threads[i]->join(); //wait for thread to exit
        delete threads[i];
    }
    delete[] threads;
#else
    printf("Single threaded mode.\r\n");
    thread_func(y_min, y_max);
#endif

    //Analytical solution to the integral
    double x_integral = (1.0 / 3.0) * (x_max * x_max * x_max - x_min * x_min * x_min);
    double analytical_integral = x_integral * 0.5 * (y_max * y_max - y_min * y_min);

    printf("Numerical Integral:  %f\r\n", integral);
    printf("Analytical Integral: %f\r\n\r\n", analytical_integral);
    printf("y_counter_atomic = %d\r\n", y_counter_atomic.load());
    printf("y_counter_nonatomic = %d\r\n", y_counter_nonatomic);

    printf("Success!\r\n");

    return 0;
}