#include <iostream>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>


Eigen::MatrixXf test_dist_loop(int SIZE)
{

    Eigen::VectorXf j = Eigen::VectorXf::Random(SIZE);

    // maybe i have to construct return matrix myself explicitly?
    // Eigen::ArrayXf mat(5,5);
	
    // this works start
    Eigen::MatrixXf mat(SIZE, SIZE);
    // Eigen::MatrixXf mat2(5,5);



    // mat = ((j.transpose() * j * -2).colwise() + j.colwise().squaredNorm().transpose()).rowwise() + j.colwise().squaredNorm();
    

    // ** replicating
    // replicate could be a thing, would mean i have to replicate vector once row-vise, once column wise
    // hm that works, not bad
    // mat = j.replicate(1,SIZE) - j.transpose().replicate(SIZE,1);

    // is it efficient tho?
    // half of those are not needed (triangles)
    // would loop be faster?
    // but then no broadcasting possible
    // i'm not broadcasting anyways?

    // ** loop
    float res = 0;

    for (int i = 0; i < SIZE; i++) {
	for (int k = 0; k < (SIZE-i); k++) {
	    res = j(i) - j(k);
	    mat(i, k) = res;
	    mat(k,i) = -res;
		
	}
    }

    return mat;
}

Eigen::MatrixXf test_dist_2mat(int SIZE) {
    Eigen::VectorXf j = Eigen::VectorXf::Random(SIZE);
    Eigen::MatrixXf mat(SIZE, SIZE);
    Eigen::MatrixXf mat2(SIZE, SIZE);

    mat.colwise() += j;
    mat2 = mat.transpose();
    mat -= mat2;

    return mat;
}

Eigen::MatrixXf test_dist_repl(int SIZE) {
    Eigen::VectorXf j = Eigen::VectorXf::Random(SIZE);
    Eigen::MatrixXf mat(SIZE, SIZE);
    mat = j.replicate(1,SIZE) - j.transpose().replicate(SIZE,1);
    return mat;
}
    

PYBIND11_MODULE(eigen_dist_test, m) {
    m.doc() = "matrix test";
    m.def("test_dist_loop", &test_dist_loop, "loop test");
    m.def("test_dist_2mat", &test_dist_2mat, "copying mat");
    m.def("test_dist_repl", &test_dist_repl, "using vector replication");
}
