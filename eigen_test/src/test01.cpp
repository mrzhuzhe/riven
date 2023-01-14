#include <iostream>
#include <Eigen/Dense>
 
using namespace std;
using namespace Eigen;
int main()
{
  /*
  //  Eigen::Matrix2d mat;
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
  */
  Eigen::MatrixXd my_matrix(2,2);
  my_matrix(0,0) = 3;
  my_matrix(1,0) = 2.5;
  my_matrix(0,1) = -1;
  my_matrix(1,1) = my_matrix(1,0) + my_matrix(0,1);
  std::cout << "Here is the matrix m:\n" << my_matrix << std::endl;
  Eigen::VectorXd v(2);
  v(0) = 4;
  v(1) = v(0) - 1;
  std::cout << "Here is the vector v:\n" << v << std::endl;

}