# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <pthread.h>

long long int num_cycle = 0;
pthread_mutex_t lock;

void* estimate(void* param){
    long long int tasks = *(int*)param;
    long long int cycles = 0;
    unsigned int seed = time(NULL);

    double x, y, f1, f2, distance;
    for (int i = 0; i < tasks; i++){
	f1 = (double)rand_r(&seed) / RAND_MAX;
	x = -1 + f1 * 2;
	f2 = (double)rand_r(&seed) / RAND_MAX;
	y = -1 + f2 * 2;
	distance = x * x + y * y;
	if (distance <= 1){
	    ++cycles;
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
    double pi = 4 * num_cycle / ((double) num_toss);
    printf("%f\n", pi);

    return 0;
}
