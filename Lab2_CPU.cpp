#include <iostream>
#include <cmath>
#include <chrono>


#define pi 3.1415926535
#define N 4 //количество  коэффициентов


using namespace std;



void genFirst(double* ind, int countParents) { //Функция генерации первого(случайного) поколения

	for (int i = 0; i < countParents; i++) {
		for (int n = 0; n < N; n++) {
			if (rand() / static_cast<double>(RAND_MAX) > 0.5)
				ind[i * N + n] = rand() / static_cast<double>(RAND_MAX);
			else
				ind[i * N + n] = -rand() / static_cast<double>(RAND_MAX);
		}
	}
}

int getRandomNumber(int min, int max) // Функция для генерации равномернораспределенного случайного числа
{
	static const double fraction = 1.0 / (static_cast<double>(RAND_MAX) + 1.0);
	return static_cast<int>(rand() * fraction * (max - min + 1) + min);
}


void fitness(int count, int countInd, double* childrens, double* arrInd) { // Функция вычисления отклонения(считаем квадрат отклонения от синуса и помещаем в массив ошибок)

	double faterr = 0.0;
	double err;
	double h = 2.0 * pi / count;
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

void selectBestParents(int* indexes, double* arr, int countIndividuals, int countParents, double* bP, double* childrens) { // Функция отбора(по массиву из ошибок arr выбираем индексы и кладем "участников" отбора с этими индексами в массив bP)

	double* arr_min = (double*)malloc(countParents * sizeof(double));

	for (int p = 0; p < countParents; p++) {
		arr_min[p] = DBL_MAX;
	}

	for (int i = 0; i < countIndividuals; i++) {
		if (arr[i] < arr_min[0]) {
			arr_min[0] = arr[i];
			indexes[0] = i;
		}
	}

	for (int p = 1; p < countParents; p++) {
		for (int i = 0; i < countIndividuals; i++) {
			if (arr[i] > arr_min[p - 1] && arr[i] < arr_min[p]) {
				arr_min[p] = arr[i];
				indexes[p] = i;
			}
		}
	}


	for (int i = 0; i < countParents; i++) {
		for (int j = 0; j < N; j++) {
			bP[i * N + j] = childrens[indexes[i] * N + j];
		}

	}

}

void crossover(double* parent1, double* parent2, double* child) { //Функция скрещевания родителей

	int n = getRandomNumber(0, N);
	for (int i = 0; i < n; i++) {
		child[i] = parent1[i];
	}

	for (int i = n; i < N; i++) {
		child[i] = parent2[i];
	}

}

void mutation(double* child, double Em, double Dm) { // Функция мутации

	int n = getRandomNumber(0, N);
	double d = Dm * std::rand() / 32767.0;
	double m = Em;
	if (rand() / 32767.0 > 0.5)
		m += d;
	else
		m -= d;

	if (rand() / 32767.0 > 0.5)
		child[n] += m;
	else
		child[n] -= m;
}

void makeChildren(double* parents, double* childrens, int countParents, int countInd, double Em, double Dm) { //Функция создания ребенка на основе скрещивания и случайной мутации

	double* child;
	double* p1;
	double* p2;

	p1 = (double*)malloc(N * sizeof(double));
	p2 = (double*)malloc(N * sizeof(double));
	child = (double*)malloc(N * sizeof(double));

	for (int i = 0; i < countParents; i++) {// Выбираем родителей для будущих детей
		int l = 0;
		for (int k = 0; k < N; k++) {
			p1[k] = parents[i * N + k];
			p2[k] = parents[(countParents - i) * N + k];
			l++;
		}
		int temp = countInd / countParents;
		for (int j = 0; j < temp; j++) {

			crossover(p1, p2, child);

			if (rand() / 32767.0 > 0.5) {
				mutation(child, Em, Dm);
			}

			for (int k = 0; k < N; k++) {
				childrens[i*temp + j * N + k] = child[k];
			}
		}
	}
}



int main()
{
	int count, countInd, countParents, maxIter, maxConstIter;
	double Em, Dm;

	cout << "Enter count of points (500 - 1000): " << endl;
	cin >> count;

	cout << "Enter count of individuals (1000 - 2000): " << endl;
	cin >> countInd;

	cout << "Enter mean for Mutation: " << endl;
	cin >> Em;

	cout << "Enter varience for Mutation: " << endl;
	cin >> Dm;

	cout << "Enter count pair of parents: " << endl;
	cin >> countParents;

	countParents *= 2;

	cout << "Enter maximal count of epochs: " <<endl;
	cin >> maxIter;

	cout << "Enter maximal count of epochs with same results: " << endl;
	cin >> maxConstIter;


	double* bP = (double*)malloc(countParents * N * sizeof(double));

	genFirst(bP, countParents); // Создаем первое поколение


	int* indexes = (int*)malloc(countParents * sizeof(int));
	double* arrInd = (double*)malloc(countInd * sizeof(double));
	double* childrens = (double*)malloc(countInd * N * sizeof(double));

	double min = DBL_MAX, val = DBL_MAX;
	int indBest, sameIter = 1;
	int sumTime = 0;
	
	for (int epoch = 1; epoch <= maxIter; epoch++) { //Начинаем эволюцию!!!!

		auto begin = chrono::steady_clock::now();

		makeChildren(bP, childrens, countParents, countInd, Em, Dm);

		fitness(count, countInd, childrens, arrInd);

		selectBestParents(indexes, arrInd, countInd, countParents, bP, childrens);

		auto end = chrono::steady_clock::now();
		auto elapsed_ms = chrono::duration_cast<chrono::milliseconds>(end - begin);
		sumTime += elapsed_ms.count();
		cout << epoch << endl;
		if (arrInd[indexes[0]] < min) {
			min = arrInd[indexes[0]];
			indBest = epoch;
		}

		if (val == arrInd[indexes[0]]) {
			sameIter++;
		}

		else {
			val = arrInd[indexes[0]];
			sameIter = 1;
		}

		if (sameIter >= maxConstIter) {
			cout << "Same " << maxConstIter << " iterations" << endl;
			break;
		}
	}


	cout << "time: " << sumTime << endl;
    cout << "min: " << min << endl << "epoch: " << indBest << endl;

	//double* temp = (double*)malloc(N * sizeof(double));
	cout << "Results: " << endl;
	for (int j = 0; j < N; j++) {
		cout << bP[j] << " ";
		
	}
	cout <<endl;
	//free(bP);

}


