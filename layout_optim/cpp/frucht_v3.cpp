#include <iostream>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

/**
   calculate dists
   first get center dists, then refine those where distances are under some threshold involving node dimensions

 */


// how to do delta nodes

Eigen::MatrixXf dists1d(Eigen::VectorXf v) {
    int SIZE = v.rows();

    Eigen::MatrixXf mat(SIZE, SIZE);
    mat = v.transpose().replicate(SIZE, 1).colwise() - v;
    return mat;
}

Eigen::MatrixXf dists(Eigen::MatrixXf pts_pos){
    // D = ( (p0.transpose() * p1 * -2
    //   ).colwise() + p0.colwise().squaredNorm().transpose()
    // ).rowwise() + p1.colwise().squaredNorm();


    Eigen::MatrixXf D(pts_pos.rows(),2);
    
    std::cout << "\ncoords: \n" << pts_pos;
    std::cout << "\ntransposed: " << pts_pos.transpose();
    

    Eigen::VectorXf vec_par1;
    vec_par1 = pts_pos.rowwise().squaredNorm().transpose();
    std::cout << "\nsome vector part: " << vec_par1;

    D = ((pts_pos * pts_pos.transpose() *  -2
	).colwise() + pts_pos.rowwise().squaredNorm()
	).rowwise() + pts_pos.rowwise().squaredNorm().transpose();

    // std::cout << "\ndist mat in c: \n" << D.array().sqrt();
    // return D.array().sqrt();
    return D.cwiseSqrt();
}

PYBIND11_MODULE(frucht_v3, m) {
  m.doc() = "aasdf";
  m.def("dists", &dists, "asdf");
}
