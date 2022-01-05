#include<stdio.h>
#include<pthread.h>
#include<unistd.h>
#include<limits.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>


#define NUM_ATOMS 81

extern __global__ void coulombMatrix(double *pos, double *col, int *chargeptr, int nx, int ny, int cutoff, double bc);
extern __global__ void coulombMatrixLT(double *col, int nx, int ny);
extern __global__ void jacobi(double *arr_ptr, int *pair_arr, int n, int *cont, double tolerance);
extern __global__ void copysubmat(double *subm, int N, int num);
extern __global__ void submatrix(double *col, int nx, int *submatsizes);

pthread_mutex_t cond_var_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_var = PTHREAD_COND_INITIALIZER;
pthread_mutex_t crit_lock = PTHREAD_MUTEX_INITIALIZER;

char *frame_bufs[10] = {0,};
unsigned int frame_size;
unsigned int line1_size;
unsigned int line2_size;
unsigned int line3_size;

#define NTHREADS 3

//cudaStream_t streams[NTHREADS];

void *main_job_cuda(void *count)
{

// parser for each frame
    double frame[NUM_ATOMS][3];
    unsigned int j = 0;
    char temp1[25], temp2[25], temp3[25];
    unsigned int cnt = *(int *)count;
    char *buf;
    unsigned int i;
    int den_1, den_2, den_3;
    cudaError_t status;
    double *posptr;
    double col[NUM_ATOMS][NUM_ATOMS] = {0,};
    double *colptr, *bufptr, *submat1;
    int *chargptr;
    int charge[3] = {8, 1, 1};
    int charge_arr[NUM_ATOMS] = {0,};
    //int i,j;
    int rcut = 5;
    int dimx = 12;
    int dimy = 12;
    int nx = NUM_ATOMS;
    int ny = NUM_ATOMS;
    int submatsize[NUM_ATOMS] = {0,};
    int *submat;
    dim3 block(dimx, dimy);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);
    double bc = 9.3214112017424995;
    int n, cont=1;
    //int *pair = (int*)malloc(n*sizeof(int));
    double tolerance = 0.000000000001;
    int dim;
    int *d_cont;
 
    //if(cnt == 0)
    //    return NULL;
  
    den_1 = 0;
    den_2 = 0;
    den_3 = 0;
    //fflush(stdout);
    //printf(" val %d ", *(int *)count);
    printf("\n main job thread # %d", cnt);
    //cudaStreamCreate(&streams[cnt]);
    buf = frame_bufs[0] + cnt * frame_size + line1_size + line2_size;
    //i += line1_size + line2_size;
    while(j < NUM_ATOMS){
        memcpy(temp1, buf + 4, 25);
        memcpy(temp2, buf + 30, 25);
        memcpy(temp3, buf + 56, 25);
        //for(i = 0; i < 18; i++)
        //    printf(" %c", temp3[i]);
        //printf("\n");
	if(temp1[20] == 'E')
	    den_1 = temp1[24] - '0';

        if(temp2[20] == 'E')
	    den_2 = temp2[24] - '0';

        if(temp3[20] == 'E')
	    den_3 = temp3[24] - '0';
	    
        sscanf(temp1, "%lf", &frame[j][0]);
        sscanf(temp2, "%lf", &frame[j][1]);
        sscanf(temp3, "%lf", &frame[j][2]);
 
        if(!den_1)
	    frame[j][0] /= pow(10, den_1);

        if(!den_2)
	    frame[j][1] /= pow(10, den_2);

        if(!den_3)
	    frame[j][2] /= pow(10, den_3);

        j += 1;
        buf += line3_size;
    }

#if 1
    if(cnt == 49){
    for(j = 0; j < NUM_ATOMS; j++){
        for(i = 0; i < 3; i++){
            printf(" %.17g", frame[j][i]);
            }
        printf("\n");
    }}
#endif

// coloumb matrix creation
    j = 0;
    for(i=0; i<(sizeof(charge_arr)/sizeof(int)); i++)
    {
        charge_arr[i] = charge[j];
        j++;
        if(j == 3)
            j=0;
    }
    for(i=0; i<(sizeof(charge_arr)/sizeof(int)); i++)
    {
        printf(" %d", charge_arr[i]);
    }

    status = cudaMalloc((double **)&posptr, NUM_ATOMS*3*sizeof(double));
    //status = cudaMalloc((double **)&posptr, NUM_ATOMS*3*sizeof(double),stream[cnt]);
    if( status != cudaSuccess) {
        fprintf(stderr, " Could not allocate memory on the device!!");
    }

    status = cudaMalloc((double **)&colptr, NUM_ATOMS*NUM_ATOMS*sizeof(double));
    //status = cudaMalloc((double **)&colptr, NUM_ATOMS*NUM_ATOMS*sizeof(double),stream[cnt]);
    if( status != cudaSuccess) {
        fprintf(stderr, " Could not allocate memory on the device!!");
    }

    status = cudaMalloc((int **)&chargptr, NUM_ATOMS*sizeof(int));
    //status = cudaMalloc((int **)&chargptr, NUM_ATOMS*sizeof(int),stream[cnt]);
    if( status != cudaSuccess) {
        fprintf(stderr, " Could not allocate memory on the device!!");
    }

    status = cudaMalloc((int **)&submat, NUM_ATOMS*sizeof(int));
    //status = cudaMalloc((int **)&submat, NUM_ATOMS*sizeof(int),stream[cnt]);
    if( status != cudaSuccess) {
        fprintf(stderr, " Could not allocate memory on the device!!");
    }

    status = cudaMemcpy(posptr, frame, NUM_ATOMS*3*sizeof(double), cudaMemcpyHostToDevice);
    //status = cudaMemcpy(posptr, frame, NUM_ATOMS*3*sizeof(double), cudaMemcpyHostToDevice,stream[cnt]);
    if( status != cudaSuccess) {
        fprintf(stderr, " Could not copy the position array to the device!!");
    }

    status = cudaMemcpy(colptr, col, NUM_ATOMS*NUM_ATOMS*sizeof(double), cudaMemcpyHostToDevice);
    //status = cudaMemcpy(colptr, col, NUM_ATOMS*NUM_ATOMS*sizeof(double), cudaMemcpyHostToDevice,stream[cnt]);
    if( status != cudaSuccess) {
        fprintf(stderr, " Could not copy the position array to the device!!");
    }

    status = cudaMemcpy(chargptr, charge_arr, NUM_ATOMS*sizeof(int), cudaMemcpyHostToDevice);
    //status = cudaMemcpy(chargptr, charge_arr, NUM_ATOMS*sizeof(int), cudaMemcpyHostToDevice,stream[cnt]);
    if( status != cudaSuccess) {
        fprintf(stderr, " Could not copy the position array to the device!!");
    }

    coulombMatrix <<< grid, block >>>(posptr, colptr, chargptr, nx, ny, rcut, bc);
    //coulombMatrix <<< grid, block, 0, streams[cnt] >>>(posptr, colptr, chargptr, nx, ny, rcut, bc);
    cudaDeviceSynchronize();
    //cudaStreamSynchronize(0);
    coulombMatrixLT <<< grid, block >>>(colptr, nx, ny);
    //coulombMatrixLT <<< grid, block, 0, streams[cnt] >>>(colptr, nx, ny);
    cudaDeviceSynchronize();
    //cudaStreamSynchronize(0);

    status = cudaMemcpy(col, colptr, NUM_ATOMS*NUM_ATOMS*sizeof(double), cudaMemcpyDeviceToHost);
    //status = cudaMemcpy(col, colptr, NUM_ATOMS*NUM_ATOMS*sizeof(double), cudaMemcpyDeviceToHost, stream[cnt]);
    if( status != cudaSuccess) {
        fprintf(stderr, " Could not copy the position array to the host!!");
    }

    //if(cnt == 49){
    //print the composed coulomb matrix
    printf("\n couloumb matrix thread #%d\n", cnt);
    for(i=0; i<NUM_ATOMS; i++)
    {
        printf("\n");
        for(j=0; j<NUM_ATOMS; j++)
        {
            printf(" %lf",col[i][j]);
        }
        //printf("\n");
    }
    //}

#if 1
    pthread_mutex_lock(&crit_lock);

    status = cudaMemcpy(colptr, col, NUM_ATOMS*NUM_ATOMS*sizeof(double), cudaMemcpyHostToDevice);
    if( status != cudaSuccess) {
        fprintf(stderr, " Could not copy the coulomb matrix array to the host!!");
    }

//submatrix creation
    submatrix <<< 1, 100 >>>(colptr, nx, submat);
    //submatrix <<< 1, 100, 0, streams[cnt] >>>(colptr, nx, submat);
    cudaDeviceSynchronize();
    //cudaStreamSynchronize(0);

    status = cudaMemcpy(submatsize, submat, NUM_ATOMS*sizeof(int), cudaMemcpyDeviceToHost);
    if( status != cudaSuccess) {
        fprintf(stderr, " Could not copy the submat sizes array to the host!!");
    }

    printf("\n\n first submat half size %d\n\n", submatsize[0]);

    int num = 0;

    //pthread_mutex_lock(&crit_lock);
    printf(" comes in #%d", cnt);

    while(num < NUM_ATOMS){
        //verify the working of jacobi eigen solver for the first submatrix
        status = cudaMalloc((double **)&submat1, submatsize[num]*submatsize[num]*sizeof(double));
        if( status != cudaSuccess) {
            fprintf(stderr, " Could not allocate memory on the device!!");
        }

        bufptr = (double *)malloc(submatsize[num]*submatsize[num]*sizeof(double));
        if(!bufptr)
            fprintf(stderr, " Could not allocate memory for submatrix!!");

        copysubmat <<< 1, 1 >>>(submat1, submatsize[num], num);
	//copysubmat <<< 1, 1, 0, streams[cnt] >>>(submat1, submatsize[num], num);
        //cudaDeviceSynchronize();
	//cudaStreamSynchronize(0);

        status = cudaMemcpy(bufptr, submat1, submatsize[num]*submatsize[num]*sizeof(double), cudaMemcpyDeviceToHost);
        if( status != cudaSuccess) {
            fprintf(stderr, " Could not copy the submat sizes array to the host!!");
        }

        //printf("\n 1st submatrix \n");
        //print the 1st submatrix
        //for(i=0; i< submatsize[0]; i++){
        //    for(j=0; j<submatsize[0]; j++){
        //        printf(" %lf", *(bufptr+i*submatsize[0]+j));
        //        }
        //    printf("\n");
        //}

        cudaMalloc((void**) &d_cont, sizeof(int));
        cudaMemcpy(d_cont, &cont, sizeof(int), cudaMemcpyHostToDevice);

        dim = submatsize[num];

        if(dim % 2 == 0){
            n = dim;
        }
        else
            n = dim + 1;

        int *pair = (int*)malloc(n*sizeof(int));
        int *d_pair;
        status = cudaMalloc( (void**) &d_pair, n*sizeof(int));
        if( status != cudaSuccess) {
            fprintf(stderr, " Could not allocate pair memory on the device!!");
        }

        if(dim % 2 == 0){
            /*initializing pair matrix*/
            for (i = 0; i < n; i++)
                *(pair + i) = i;
        }
        else{
            for (i = 0; i < dim; i++)
                *(pair + i) = i;
            *(pair + n - 1) = 999;
        }
        printf("\n n %d\n", n);

        #if 1
        status = cudaMemcpy(submat1, bufptr, submatsize[num]*submatsize[num]*sizeof(double), cudaMemcpyHostToDevice);
        if( status != cudaSuccess) {
            fprintf(stderr, " Could not copy the submat array to the host!!");
        }

        status = cudaMemcpy(d_pair, pair, n*sizeof(int), cudaMemcpyHostToDevice);
        if( status != cudaSuccess) {
            fprintf(stderr, " Could not copy the pair array to the host!!");
        }

        jacobi<<<1, n/2>>>(submat1, d_pair, submatsize[num], d_cont, tolerance);
	//jacobi<<<1, n/2, 0, streams[cnt]>>>(submat1, d_pair, submatsize[num], d_cont, tolerance);
	//cudaDeviceSynchronize();
        //cudaStreamSynchronize(0);

        cudaMemcpy(bufptr, submat1, submatsize[num]*submatsize[num]*sizeof(double), cudaMemcpyDeviceToHost);

        printf("\n\n eigen values here for #%d", cnt);
    //print the 1st submatrix
        for(i=0; i< submatsize[num]; i++){
            for(j=0; j<submatsize[num]; j++){
	      if(i == j)
                  printf(" %lf", *(bufptr+i*submatsize[num]+j));
              }
            printf("\n");
        }
        num += 1;
        cudaFree(d_pair);
        cudaFree(d_cont);
        cudaFree(submat1);
        free(bufptr);
        free(pair);
    #endif
    }

    pthread_mutex_unlock(&crit_lock);

#endif    
    cudaFree(posptr);
    cudaFree(colptr);
    cudaFree(chargptr);

}

void *parse_pos_file(void *arg)
{
    unsigned int i = 0;
    //char *buf;
    int count = 0;
    pthread_t threads[NTHREADS];
    void * retvals[NTHREADS];
    int *cnt;

    printf(" Thread 2");
    // cond_wait
    pthread_mutex_lock(&cond_var_lock);
    while(frame_bufs[0] == 0)
        pthread_cond_wait(&cond_var, &cond_var_lock);
    pthread_mutex_unlock(&cond_var_lock);

    //printf("\n comes here --1");
    //printf("\n");
    //buf = frame_bufs[0];
    //for(i = 0; i < 100; i++)
    //    printf("%c", *(buf+i));

   
    // launch 50 threads which parses the pos frame buffer and performs cuda operations in parallel.
    // each of the thread creates the coulomb matrix, does submatrix reductions and computes eigen values.
    // the eigen values are finally written as hdf5 files.
//    int k = 0;
    for(count = 0; count < NTHREADS; count++)
    {
        fflush(stdout);
        cnt = (int *)malloc(1*sizeof(int));
        *cnt = count;
        if(pthread_create(&threads[count], NULL, main_job_cuda, (void *)cnt) != 0)
	{
	    printf("error: cannot create thread # %d\n", *cnt);
	    return (void *)NULL;
	}
    }

    for(i = 0; i < NTHREADS; i++)
    {
        if(pthread_join(threads[i], &retvals[i]) != 0)
	{
	    printf("error: cannot join thread # %d\n", i);
	    return (void *)NULL;
	}
    }

#if 0
// parser for each frame
    double frame[NUM_ATOMS][3];
    unsigned int j = 0;
    char temp1[18], temp2[18], temp3[18];
    buf = frame_bufs[0] + line1_size + line2_size;
    //i += line1_size + line2_size;
    while(j < NUM_ATOMS){
        memcpy(temp1, buf + 6, 18);
	memcpy(temp2, buf + 32, 18);
	memcpy(temp3, buf + 58, 18);
	for(i = 0; i < 18; i++)
	    printf(" %c", temp3[i]);
        printf("\n");
        sscanf(temp1, "%lf", &frame[j][0]);
	sscanf(temp2, "%lf", &frame[j][1]);
	sscanf(temp3, "%lf", &frame[j][2]);
	j += 1;
	buf += line3_size;
    }
#endif 

#if 0
    for(j = 0; j < NUM_ATOMS; j++){
        for(i = 0; i < 3; i++){
            printf(" %.17g", frame[j][i]);
	    }
	printf("\n");
    }
#endif

}

void *read_pos_file(void *pth)
{
    char *path = (char *)pth;
    FILE *fp;
    char buf[256] = {0,};
    unsigned int i = 0;
    //unsigned int frame_size = 0;
    char *buff;
    printf("\n File path is %s", path);

    fp = fopen(path, "r");

    // estimate the memory size needed for a frame from pos file. Reads the first three lines.
    while(fgets(buf, 256, (FILE *)fp) != NULL){
        //printf("strlen(buf) is %d", strlen(buf));
	i++;

	if(i == 1){
	    frame_size += strlen(buf);
	    line1_size = strlen(buf);
	}
	else if(i == 2){
	frame_size += strlen(buf);
	line2_size = strlen(buf);
	}
	else{
	    frame_size += NUM_ATOMS * strlen(buf);
	    line3_size = strlen(buf);
        }
	if(i == 3)
	    break;
    }

#if 1
    for(i = 0; i < 10; i++)
        printf("%c", buf[i]);
#endif

//    fclose(fp);
    printf("memory size req for a frame is %d", frame_size); 
    fseek(fp, 0, SEEK_SET);

    pthread_mutex_lock(&cond_var_lock);
    buff = (char *)malloc(frame_size * 50 * sizeof(char));
    fread(buff, sizeof(char), frame_size * 50, fp);
    frame_bufs[0] = buff;
    pthread_cond_signal(&cond_var);
    pthread_mutex_unlock(&cond_var_lock);

    for(i = 1; i < 10; i++)
    {
        buff = (char *)malloc(frame_size * 50 * sizeof(char));
	fread(buff, sizeof(char), frame_size * 50, fp);
        frame_bufs[i] = buff;
    }
    fclose(fp);
}


int main(int argc, char *argv[])
{
    pthread_t thread1, thread2;
    int ret1, ret2; 
    char pos_file_name[] = "pos.xyz";
    char cwd[PATH_MAX];
    char file_path[PATH_MAX + strlen(pos_file_name)];

    if(argc == 1)
    {    
        printf("Program expects the directory name where pos/frc/ener file is located");
        return -1;
    }

    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("Current working dir: %s\n", cwd);
    } else {
       perror("getcwd() error");
       return 1;
    }

    //printf("cwd is %s", cwd);
    strcat(file_path, cwd);
    strcat(file_path, "/");
    strcat(file_path, argv[1]);
    strcat(file_path, "/");
    strcat(file_path, pos_file_name);
    printf("\n file_path is %s", file_path);

// launch the thread which reads data from the pos file and stores in small buffers.
    ret1 = pthread_create( &thread1, NULL, read_pos_file, (void *)file_path);     
    ret2 = pthread_create( &thread2, NULL, parse_pos_file, (void *)NULL);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    return 0;
}
