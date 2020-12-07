# include <stdio.h>
# include <stdlib.h>
# include <mpi.h>

# define MASTER 0
# define FROM_MASTER 1
# define FROM_WORKER 2

// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from stdin
//
// n_ptr:     pointer to n
// m_ptr:     pointer to m
// l_ptr:     pointer to l
// a_mat_ptr: pointer to matrix a (a should be a continuous memory space for placing n * m elements of int)
// b_mat_ptr: pointer to matrix b (b should be a continuous memory space for placing m * l elements of int)
void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
	         	int **a_mat_ptr, int **b_mat_ptr){
    int size, rank;
    int *ptr;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == MASTER){
	scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
	int n = *n_ptr, m = *m_ptr, l = *l_ptr;
    	// printf("%d %d %d\n", n, m, l);
	*a_mat_ptr = (int*)malloc(sizeof(int) * n * m);
	*b_mat_ptr = (int*)malloc(sizeof(int) * m * l);

	for (int i = 0; i < n; i++){
	    for (int j = 0; j < m; j++){
		ptr = *a_mat_ptr + i * m + j;
		scanf("%d", ptr);
	    }
	}

	for (int i = 0; i < m; i++){
	    for (int j = 0; j < l; j++){
		ptr = *b_mat_ptr + i * l + j;
		scanf("%d", ptr);
	    }
	}

	/* debug purpose */
	// for (int i = 0; i < n; i++){
	//     for (int j = 0; j < m; j++){
	// 	ptr = *a_mat_ptr + i * m + j;
	// 	printf("%d ", *ptr);
	//     }
	//     printf("\n");
	// }

	// for (int i = 0; i < m; i++){
	//     for (int j = 0; j < l; j++){
	// 	ptr = *b_mat_ptr + i * l + j;
	// 	printf("%d ", *ptr);
	//     }
	//     printf("\n");
	// }
    }
}

// Just matrix multiplication (your should output the result in this function)
//
// n:     row number of matrix a
// m:     col number of matrix a / row number of matrix b
// l:     col number of matrix b
// a_mat: a continuous memory placing n * m elements of int
// b_mat: a continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat){
    int size, rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int numworker, source, dest, mtype, rows, averow, extra, offset;
    int N, M, L;
    int i, j, k;
    numworker = size - 1;
    if (rank == MASTER){
	int *c;
    	c = (int*)malloc(sizeof(int) * n * l);
        /* Send matrix data to the worker tasks */
        averow = n / numworker;
        extra = n % numworker;
        offset = 0;
        mtype = FROM_MASTER;
        for (dest = 1; dest <= numworker; dest++){
            MPI_Send(&n, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&m, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&l, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            rows = (dest <= extra)? averow + 1: averow;
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
	    // printf("master send info to rank %d\n", dest);
            MPI_Send(&a_mat[offset * m], rows * m, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&b_mat[0], m * l, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            offset += rows;
        }
        /* Receive results from worker tasks */
        mtype = FROM_WORKER;
        for (i = 1; i <= numworker; i++){
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset * l], rows * l, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
	    // printf("master receive from %d\n", source);
        }
        /* Print results */
        for (i = 0; i < n; i++){
            for (j = 0; j < l; j++){
        	printf("%d", c[i * l + j]);
		if (j != l-1) printf(" ");
            }
            printf("\n");
        }
	free(c);
    }
    if (rank > MASTER){
        mtype = FROM_MASTER;
        MPI_Recv(&N, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&M, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&L, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
	int *a;
    	int *b;
    	int *c;
    	a = (int*)malloc(sizeof(int) * N * M);
    	b = (int*)malloc(sizeof(int) * M * L);
    	c = (int*)malloc(sizeof(int) * N * L);
	// printf("n: %d, m: %d, l: %d\n", N, M, L);
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
	// printf("rank %d receive from master\n", rank);
        MPI_Recv(&a[0], rows * M, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&b[0], M * L, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
	// printf("\n");
	// for (int i = 0; i < rows * M; i++) printf("a[%d]: %d\n", i, a[i]);
	// for (int i = 0; i < L * M; i++) printf("b[%d]: %d\n", i, b[i]);
	// printf("\n");

        for (k = 0; k < L; k++){
            for (i = 0; i < rows; i++){
        	c[i * L + k] = 0;
        	for (j = 0; j < M; j++){
        	    c[i * L + k] += a[i * M + j] * b[j * L + k];
		    // printf("a[%d][%d] = %d\n", i, j, a[i*M + j]);
		    // printf("b[%d][%d] = %d\n", j, k, b[j*L + k]);
        	}
            }
        }

        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&c[0], rows * L, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
	// printf("rank %d send result\n", rank);
	free(a);
    	free(b);
	free(c);
    }
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat){
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == MASTER){
	free(a_mat);
	free(b_mat);
    }
}
