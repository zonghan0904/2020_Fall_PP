# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <pthread.h>
# include <immintrin.h>

# include "simdxorshift128plus.h"

volatile long long int num_cycle = 0;
pthread_mutex_t lock;

void* estimate(void* param){
    long long int tasks = *(int*)param;
    long long int cycles = 0;
    avx_xorshift128plus_key_t mykey;
    avx_xorshift128plus_init(324, 4444, &mykey);
    __m256 full = _mm256_set_ps(INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX);

    for (int i = 0; i < tasks; i+=8){
	__m256i x_i = avx_xorshift128plus(&mykey);
	__m256 x_f = _mm256_cvtepi32_ps(x_i);
	__m256 x = _mm256_div_ps(x_f, full);

	__m256i y_i = avx_xorshift128plus(&mykey);
	__m256 y_f = _mm256_cvtepi32_ps(y_i);
	__m256 y = _mm256_div_ps(y_f, full);

	__m256 x_2 = _mm256_mul_ps(x, x);
	__m256 y_2 = _mm256_mul_ps(y, y);
	__m256 sum = _mm256_add_ps(x_2, y_2);

	float val[8];
	_mm256_store_ps(val, sum);

	for (int i = 0; i < 8; i++){
	    if (val[i] <= 1.f) ++cycles;
	}
    }

    pthread_mutex_lock(&lock);
    num_cycle += cycles;
    pthread_mutex_unlock(&lock);
}

int main(int argc, char** argv){
    if (argc != 3){
	printf("usage: ./pi.out {CPU core} {Number of tosses}\n");
	return 1;
    }

    int num_thread = atoi(argv[1]);
    long long int num_toss = atoll(argv[2]);

    pthread_t* threads;
    threads = (pthread_t*)malloc(num_thread * sizeof(pthread_t));
    long long int num_task = num_toss / num_thread;
    pthread_mutex_init(&lock, NULL);

    for (int i = 0; i < num_thread; i++){
	if (i == num_thread - 1){
	    num_task += num_toss % num_thread;
	    pthread_create(&threads[i], NULL, estimate, (void*)&num_task);
	}
	else{
	    pthread_create(&threads[i], NULL, estimate, (void*)&num_task);
	}
    }

    for (int i = 0; i < num_thread; i++){
	pthread_join(threads[i], NULL);
    }

    free(threads);
    pthread_mutex_destroy(&lock);
    float pi = 4 * num_cycle / ((float) num_toss);
    printf("%lf\n", pi);

    return 0;
}
