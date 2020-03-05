#include <iostream>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>


Eigen::MatrixXf dist_loop_no_openmp(int SIZE)
{

    Eigen::VectorXf j = Eigen::VectorXf::Random(SIZE);

    Eigen::MatrixXf mat(SIZE, SIZE);

    float res = 0;
    // #pragma omp parallel for

    for (int i = 0; i < SIZE; i++) {
	for (int k = 0; k < (SIZE-i); k++) {
	    res = j(i) - j(k);
	    mat(i, k) = res;
	    mat(k,i) = -res;
		
	}
    }

    return mat;
}

Eigen::MatrixXf so_dists() {
    // Eigen::MatrixXf p0(2, 4);
    // p0 <<
    // 	1, 23, 6, 9,
    // 	3, 11, 7, 2;

    // Eigen::MatrixXf p1(2, 2);
    // p1 <<
    // 	2, 20,
    // 	3, 10;

    Eigen::MatrixXf p0(1, 4);
    p0 <<
    	1, 23, 6, 9;


    Eigen::MatrixXf p1(1, 4);
    p1 <<
	1, 23, 6, 9;
    
    Eigen::MatrixXf D(p0.cols(), p0.rows());
    Eigen::MatrixXf e1(p0.cols(), p0.rows());
    Eigen::MatrixXf e11(p0.cols(), p0.rows());
    Eigen::MatrixXf e12(p0.cols(), p0.rows());
    Eigen::MatrixXf e13(p0.cols(), p0.rows());
    Eigen::MatrixXf e2(p0.cols(), p0.rows());

    // D = ( (p0.transpose() * p1 * -2 ).colwise() + p0.colwise().squaredNorm().transpose()
    // 	).rowwise() + p1.colwise().squaredNorm();

    // e1 = ((p0.transpose() * p1 * -2 ).colwise() + p0.colwise().squaredNorm().transpose());
    // .rowwise();
    // rowwise/colwise are basically the things to which the following methods attach
    // maybe broadcast on replicated vector?

    e11 = (p0.transpose() * p1);
    e12 = p0.colwise().sum().transpose();
    // e13 = e11 + e12;

    e2 = p1.colwise().sum();
    

    // std::cout << e1;
    std::cout << e11 << std::endl;
    std::cout << e12 << std::endl;
    // std::cout << e13;
    std::cout << e2 << std::endl;

    
    // D = ((p0.transpose() * p1).colwise() + p0.colwise().sum().transpose()).rowwise() + p1.colwise().sum();


    // D = p0.transpose() * p1;
    // i don't want no matrix products tho
    // operation can't be left hand rowwise or columnwise, has to be returned in one go
    // D = p1.rowwise() - p0.transpose();

    return D;
}


Eigen::MatrixXf dist_loop_openmp(int SIZE)
{

    Eigen::VectorXf j = Eigen::VectorXf::Random(SIZE);

    Eigen::MatrixXf mat(SIZE, SIZE);

    // is it efficient tho?
    // half of those are not needed (triangles)
    // would loop be faster?
    // but then no broadcasting possible
    // i'm not broadcasting anyways?

    // ** loop
    float res = 0;
    #pragma omp parallel for

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
    // replicate could be a thing, would mean i have to replicate vector once row-vise, once column wise
    // hm that works, not bad

    Eigen::VectorXf j = Eigen::VectorXf::Random(SIZE);
    Eigen::MatrixXf mat(SIZE, SIZE);
    mat = j.replicate(1,SIZE) - j.transpose().replicate(SIZE,1);
    return mat;
}

Eigen::MatrixXf test_dist_repl2(int SIZE) {

    // rowwise/colwise are basically the things to which the following methods attach
    // maybe broadcast on replicated vector?
    // minimize transposing by replicating transposed vector row-wise and then subtracing col-vector colwise
    
    Eigen::VectorXf j = Eigen::VectorXf::Random(SIZE);
    Eigen::MatrixXf mat(SIZE, SIZE);

    
    // mat = j.replicate(1,SIZE) - j.transpose().replicate(SIZE,1);
    mat = j.transpose().replicate(SIZE,1).colwise() - j;
    
    return mat;
}



    

PYBIND11_MODULE(eigen_dist_test, m) {
    m.doc() = "matrix test";
    m.def("dist_loop_no_openmp", &dist_loop_no_openmp, "loop test");
    m.def("dist_loop_openmp", &dist_loop_openmp, "loop test");
    m.def("so_dists", &so_dists, "loop test");
    
    
    // m.def("test_dist_2mat", &test_dist_2mat, "copying mat");
    m.def("test_dist_repl", &test_dist_repl, "using vector replication");
    m.def("test_dist_repl2", &test_dist_repl2, "using vector replication");

}
