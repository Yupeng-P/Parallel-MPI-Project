/* File: mpi_vector_ops.c
 * COMP 137-1 Spring 2020
 */

// ssh yzheng@ecs-cluster.serv.pacific.edu
// mpicc -o para mpi_vector_para.c -std=c99
// srun --nodes=1 --ntasks=4 ./para
// scp .g/mpi_vector_para.c yzheng@ecs-cluster.serv.pacific.edu:./
// module load mpi/openmpi-2.0.2

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

char* infile = NULL;
char* outfile = NULL;

int readInputFile(char* filename, long* n_p, double* x_p, double** A_p, double** B_p)
{
    long i;
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) return 0;
    fscanf(fp, "%ld\n", n_p);
    fscanf(fp, "%lf\n", x_p);
    *A_p = malloc(*n_p*sizeof(double));
    *B_p = malloc(*n_p*sizeof(double));
    for (i=0; i<*n_p; i++) fscanf(fp, "%lf\n", (*A_p)+i);
    for (i=0; i<*n_p; i++) fscanf(fp, "%lf\n", (*B_p)+i);
    return 1;
}

int writeOutputFile(char* filename, long n, double y, double* C, double* D)
{
    long i;
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) return 0;
    fprintf(fp, "%ld\n", n);
    fprintf(fp, "%lf\n", y);
    for (i=0; i<n; i++) fprintf(fp, "%lf\n", C[i]);
    for (i=0; i<n; i++) fprintf(fp, "%lf\n", D[i]);
    return 1;
}

void serialSolution(long n, double x, double* A, double* B, double* y, double* C, double* D)
{
    long i;
    double dp = 0.0;
    for (i=0; i<n; i++)
    {
        C[i] = x*A[i];
        D[i] = x*B[i];
        dp += A[i]*B[i];
    }
    *y = dp;
}

int main(int argc, char* argv[])
{
    long    n=0;     /* size of input arrays */
    double  x;       /* input scalar */
    double* A;       /* input vector */
    double* B;       /* input vector */
    double* C;       /* output vector xA */
    double* D;       /* output vector xB */
    double  y;       /* output scalar A dot B */
    double* my_A;
    double* my_B;
    double* my_C;
    double* my_D; 
    double my_y;
    long my_n;
    int my_rank, comm_sz;
    double start, finish;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    start = MPI_Wtime();

    if(my_rank == 0){
        if (argc<3)
        {
            n = -1;
            fprintf(stderr, "Command line arguments are required.\n");
            fprintf(stderr, "argv[1] = name of input file\n");
            fprintf(stderr, "argv[2] = name of input file\n");
        }
        else
        {
            infile = argv[1];
            outfile = argv[2];
            if (!readInputFile(infile, &n, &x, &A, &B))
            {
                fprintf(stderr, "Error opening input files. Aborting.\n");
                n = -1;
            }
        }
        if (n < 0)
        {
            fprintf(stderr, "Aborting task due to input errors.\n");
            exit(1);
        }
    }

    MPI_Bcast(&n, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&x, 1, MPI_DOUBLE, 0 , MPI_COMM_WORLD);

    my_n = n/comm_sz;

    my_A = malloc(my_n*sizeof(double));
    my_B = malloc(my_n*sizeof(double));
    
    MPI_Scatter(A, my_n, MPI_DOUBLE, my_A, my_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, my_n, MPI_DOUBLE, my_B, my_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    C = malloc(n*sizeof(double));
    D = malloc(n*sizeof(double));

    my_C = malloc(my_n*sizeof(double));
    my_D = malloc(my_n*sizeof(double));

    serialSolution(my_n, x, my_A, my_B, &my_y, my_C, my_D);

    MPI_Reduce(&my_y, &y, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Gather(my_C, my_n, MPI_DOUBLE, C, my_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(my_C, my_n, MPI_DOUBLE, D, my_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (my_rank == 0){
        if (!writeOutputFile(outfile, n, y, C, D))
        {
            fprintf(stderr, "Error opening output file. Aborting.\n");
            exit(1);
        }

        finish = MPI_Wtime();
        printf("Elapsed time = %lf second\n",finish - start);
    }

    MPI_Finalize();

    

    free(my_A);
    free(my_B);
    free(my_C);
    free(my_D);
    free(A);
    free(B);
    free(C);
    free(D);
    return 0;
}
