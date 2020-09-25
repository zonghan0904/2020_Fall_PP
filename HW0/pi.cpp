# include <iostream>
# include <cstdlib>
# include <ctime>

using namespace std;

int main(void){
    srand(time(NULL));
    long long int num_cycle = 0, num_toss = 100000000, toss;
    double x, y, f1, f2, distance;
    for (toss = 0; toss < num_toss; toss++){
	f1 = (double)rand() / RAND_MAX;
	x = -1 + f1 * 2;
	f2 = (double)rand() / RAND_MAX;
	y = -1 + f2 * 2;
	distance = x * x + y * y;
	if (distance <= 1){
	    num_cycle++;
	}
    }
    double pi = 4 * num_cycle / ((double) num_toss);
    cout << "pi: " << pi << endl;
    return 0;
}
