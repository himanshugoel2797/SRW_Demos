#include <atomic>
#include <thread>
#include <mutex>
#include <shared_mutex>

const int iteration_count = 10;
const int thread_count = 3;

volatile int *common_data;
volatile int *out_data;

#define CPP11

std::atomic_int32_t cntr = 0;

#ifdef CPP11
std::mutex sync;
#elif CPP14
std::shared_timed_mutex sync;
#elif CPP17
std::shared_mutex sync;
#endif

void thread_func()
{
    #ifdef CPP11
    sync.lock();
    #elif CPP14
    std::unique_lock<std::shared_timed_mutex> write_lock(sync);
    #elif CPP17
    #endif
    
    for (int i = 0; i < iteration_count; i++)
    {
        common_data[i] = i;
    }
    
    #ifdef CPP11
    sync.unlock();
    #elif CPP14
    //Do nothing, lock is released when write_lock falls out of scope
    #elif CPP17            
    #endif
}

void thread_read_func(int id)
{
    
    #ifdef CPP11
    sync.lock();
    #elif CPP14
    std::shared_lock<std::shared_timed_mutex> read_lock(sync);
    #elif CPP17
    #endif
    
    for (int i = 0; i < iteration_count; i++)
    {
        out_data[i * thread_count + id] = common_data[i];
    }
    
    #ifdef CPP11
    sync.unlock();
    #elif CPP14
    //Do nothing, lock is released when read_lock falls out of scope
    #elif CPP17            
    #endif
}

int main()
{
    common_data = new int[iteration_count](); //zero initialize array of data
    out_data = new int[thread_count * iteration_count]();

    //Create threads
    std::thread writer_thread(thread_func);
    std::thread ** threads = new std::thread*[thread_count];
    for (int i = 0; i < thread_count; i++)
        threads[i] = new std::thread(thread_read_func, i); //create a thread that calls thread_read_func with parameter i

    writer_thread.join();
    for (int i = 0; i < thread_count; i++)
        threads[i]->join(); //wait for thread to exit

    //check result
    int *iter = (int*)out_data;
    for (int i = 0; i < iteration_count; i++)
    {
        int ref_val = *iter++;
        for (int j = 1; j < thread_count; j++)
        {
            if(*(iter++) != ref_val)
            {
                printf("Error, mismatch detected at %d!\r\n", i);
                return -1;
            }
        }
    }

    printf("Success!\r\n");
    return 0;
}