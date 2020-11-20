
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <stdio.h>
#include <cmath>
#include <float.h>
#include <chrono>
#include <iostream>

#define BLOCK_SIZE 16
#define pi 3.1415926535
#define N 4 


using namespace std;
using namespace thrust;



void fitness(int count, int countInd, double* childrens, double* arrInd) {

	double faterr = 0.0;
	double err;
	double h = pi / count;
	double X;

	for (int i = 0; i < countInd; i++) {

		for (int j = 0; j < count; j++) {
			err = 0.0;
			for (int k = 0; k < N; k++) {
				X = pow(j * h + h, k);
				err += childrens[i * N + k] * X;
			}
			faterr += pow(sin(j * h + h) - err, 2);
		}

		arrInd[i] = faterr;
		faterr = 0.0;
	}


}

void selectBestParents(int* indexes, double* arrInd, int countInd, int countParents, double* bP, double*childrens) {

	for (int i = 0; i < countInd; i++) {
		indexes[i] = i;
	}

	sort_by_key(arrInd, arrInd + countInd, indexes);

	for (int i = 0; i < countParents; i++) {
		for (int j = 0; j < N; j++) {
			bP[i * N + j] = childrens[N * indexes[i] + j];
		}
	}

}


__global__ void MakeChildrens_GPU(int Em, int Dm, int count, int countInd, double* d_bP, double* d_arrInd, double* d_childrens) {

	int countParents = 10;

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	if (index < countParents) {
		for (int i = index; i < countParents; i += stride) {
			int countChildren = countInd / countParents;
			for (int j = 0; j < countChildren; j++) {

				curandState state;
				curand_init((unsigned long long)clock() + index, 0, 0, &state);

				int n = floor(curand_uniform_double(&state) * N);

				

				for (int k = 0; k < n; k++) {
					d_childrens[i * countChildren * N + j * N + k] = d_bP[i * N + k];
				}

				for (int k = n; k < N; k++) {
					d_childrens[i * countChildren * N + j * N + k] = d_bP[(countParents - i) * N + k];
				}
				curand_init((unsigned long long)clock() + index, 0, 0, &state);

				if (curand_uniform_double(&state) > 0.5) {

					double d = Dm * curand_uniform_double(&state);
					double m = Em;
					if (curand_uniform_double(&state) > 0.5)
						m += d;
					else
						m -= d;
					int nn = (int)(curand_uniform_double(&state) * N);
					if (curand_uniform_double(&state) > 0.5)
						d_childrens[i * countChildren * N + j * N + nn] += m;
					else
						d_childrens[i * countChildren * N + j * N + nn] -= m;
				}
			}
		}
	}

}


int main()
{

	int count, countInd, maxIter, maxConstIter;
	double Em, Dm;
	

	int countParents = 10;

	cout << "Enter count of points (500 - 1000): " << endl;
	cin >> count;

	cout << "Enter count of individuals (1000 - 2000): " << endl;
	cin >> countInd;

	cout << "Enter mean for Mutation: " << endl;
	cin >> Em;

	cout << "Enter varience for Mutation: " << endl;
	cin >> Dm;

	cout << "Enter maximal count of epochs: " << endl;
	cin >> maxIter;

	cout << "Enter maximal count of epochs with same results: " << endl;
	cin >> maxConstIter;

	


	double* h_bP = new double[countParents * N];
	for (int i = 0; i < countParents; i++) {
		for (int j = 0; j < N; j++) {
			h_bP[i * N + j] = 0.0;
		}
	}

	
	int* h_indexes = new int[countInd];
	double* h_arrInd = new double[countInd];
	double* h_childrens = new double[countInd * N];



	for (int i = 0; i < countInd; i++) {
		for (int j = 0; j < N; j++) {
			h_childrens[i * N + j] = 0.0;
		}
	}


	int* d_indexes;
	double* d_arrInd;
	double* d_bP;
	double* d_childrens;
	double min = DBL_MAX, val = DBL_MAX;
	int sameIter = 1;


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float gpu_elapsed_time_ms;
	float sumTime = 0.0;


	cudaMalloc((void**)&d_arrInd, countInd * sizeof(double));
	cudaMalloc((void**)&d_bP, countParents * N * sizeof(double));
	cudaMalloc((void**)&d_childrens, countInd * N * sizeof(double));


	int epoch;
	for (epoch = 1; epoch <= maxIter; epoch++) {


		cudaEventRecord(start, 0);
		cudaMemcpy(d_bP, h_bP, countParents * N * sizeof(double), cudaMemcpyHostToDevice);

		MakeChildrens_GPU << <countParents * countInd, 1 >> > (Em, Dm, count, countInd, d_bP, d_arrInd, d_childrens);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);

		sumTime += gpu_elapsed_time_ms;


		cudaMemcpy(h_childrens, d_childrens, countInd * N * sizeof(double), cudaMemcpyDeviceToHost);
		auto begin = chrono::steady_clock::now();

		fitness(count, countInd, h_childrens, h_arrInd);
		selectBestParents(h_indexes, h_arrInd, countInd, countParents, h_bP, h_childrens);

		auto end = chrono::steady_clock::now();

		auto elapsed_ms = chrono::duration_cast<chrono::milliseconds>(end - begin);

		sumTime += elapsed_ms.count();


		if (h_arrInd[0] < min) {
			min = h_arrInd[0];
		}

		if (val == h_arrInd[0]) sameIter++;

		else {
			val = h_arrInd[0];
			sameIter = 1;
		}

		if (sameIter >= maxConstIter) {
			break;
		}

	}


	cout<< "Time: " << sumTime << endl;

	cout << "min: " << min << endl << "epochs: " << epoch << endl;

	double* temp = (double*)malloc(N * sizeof(double));
	cout << "Result: " << endl;
	for (int j = 0; j < N; j++) {
		cout << h_bP[j] << " ";
	}
	cout << endl;

	cudaFree(d_arrInd);
	cudaFree(d_bP);
	cudaFree(d_childrens);


}







