#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    long long int num_task = tosses / world_size;
    int SEED = 78;
    int tag = 0;
    long long int count = 0;

    int seed = world_rank * SEED, dest = 0;
    long long int local_cnt = 0;
    srand(seed);
    for (long long int toss = 0; toss < num_task; toss++){
        double x = (double) rand() / RAND_MAX;
        double y = (double) rand() / RAND_MAX;
        double distance = x * x + y * y;
        if (distance <= 1){
	   local_cnt++;
        }
    }

    if (world_rank > 0)
    {
        // TODO: MPI workers
	int dest = 0;
	MPI_Request req;
	MPI_Isend(&local_cnt, 1, MPI_UNSIGNED_LONG, dest, tag, MPI_COMM_WORLD, &req);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
	int num_request = world_size - 1;
	MPI_Status status[num_request];
        MPI_Request requests[num_request];
	long long int buf[num_request];
	for (int source = 1; source < world_size; source++){
	    MPI_Irecv(&buf[source-1], 1, MPI_UNSIGNED_LONG, source, tag, MPI_COMM_WORLD, &requests[source-1]);
	}
        MPI_Waitall(num_request, requests, status);
	for (int i = 0; i < num_request; i++){
	    local_cnt += buf[i];
	}
    }

    if (world_rank == 0)
    {
        // TODO: PI result
	count = local_cnt;
	pi_result = 4 * count / (double) tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
