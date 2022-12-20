#include <iostream>
#include <Eigen/Dense>
 
using namespace std;
using namespace Eigen;
int main()
{
  //Eigen::Matrix2d mat;
  MatrixXd mat(2,2);

  mat << 1, 2,
         3, 4;
 
  //cout << "" << mat.coeff()       << endl;
  cout << "Here is mat.sum():       " << mat.sum()       << endl;
  cout << "Here is mat.prod():      " << mat.prod()      << endl;
  cout << "Here is mat.mean():      " << mat.mean()      << endl;
  cout << "Here is mat.minCoeff():  " << mat.minCoeff()  << endl;
  cout << "Here is mat.maxCoeff():  " << mat.maxCoeff()  << endl;
  cout << "Here is mat.trace():     " << mat.trace()     << endl;
}