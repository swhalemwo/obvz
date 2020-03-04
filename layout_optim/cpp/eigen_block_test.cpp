#include <iostream>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// #include <list>

/** 
    get matrix blocks with eigen blocks, unoptimized

 */
Eigen::MatrixXf blocker1 (int SIZE) {

    // Eigen::MatrixXf mat(4,4);
    // mat << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;

    Eigen::MatrixXf mat = Eigen::MatrixXf::Random(SIZE*2,SIZE*2);
    Eigen::MatrixXf res_mat = Eigen::MatrixXf::Random(SIZE,SIZE);
    
    // #pragma omp parallel for
    for (int i=0; i<SIZE; i++){
	for (int k=0; k<SIZE-i; k++) {
	    res_mat(i,k) = mat.block<2,2>(i*2,k*2).maxCoeff();
	}
    }
    // std::cout << mat.block<2,2>(0,0);
    // std::cout << mat;

    return res_mat;
}

/**
   check how to get block-reduced array without looping over blocks
*/
void blocker3 () {


    Eigen::MatrixXi M1(4,4);    // Column-major storage
    M1 << 1, 2, 3,
	4, 5, 6,
	7, 8, 9, 10, 11,12,13,14,15,16;

    std::cout << "M1" << std::endl << M1 << std::endl << std::endl;


    Eigen::Map<Eigen::RowVectorXi> v1(M1.data(), M1.size());
    std::cout << "v1:" << std::endl << v1 << std::endl << std::endl;
    
    Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> M2(M1);

    std::cout << "M2:" << std::endl << M2 << std::endl << std::endl;

    
    
    // Eigen::Map<Eigen::RowVectorXf> v2(M2.data(), M2.size());

    // Eigen::Map<Eigen::MatrixXi> v2(M2.data(), 8, 2);
    Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> v2(M1.data(), 8, 2);


    // Eigen::Map<Eigen::MatrixXf<Eigen::Dynamic,Eigen::Dynamic> > v2(M2.data());
    // Eigen::Map<Eigen::Matrix<int, 8, 2, Eigen::RowMajor> > v2(M2.data());
    std::cout << "v2:" << std::endl << v2 << std::endl;
    // std::cout << Eigen::Map < Eigen::Matrix<int,8,2> > (M1);

    // int array[8];
    // for(int i = 0; i < 8; ++i) array[i] = i;
    // std::cout << "Row-major:\n" << Eigen::Map<Eigen::Matrix<int,4,2, Eigen::RowMajor> >(array) << std::endl;

    std::cout << "next try :\n" << Eigen::Map<Eigen::Matrix<int,8,2, Eigen::RowMajor> >(M2.data()) << std::endl;

    
    // works but needs to transpose
}


// Eigen::PermutationMatrix<Dynamic> create_perm(int row_order[]) {

//void create_perm(pybind11::list row_order) {

// void create_perm(std::list<int> row_order) {
// Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic, int> create_perm(Eigen::VectorXi row_order) {
    
// void create_perm(const std::vector<int> &v) {
// Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> create_perm(Eigen::VectorXi row_order) {



Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> create_perm(Eigen::VectorXi row_order) {    

    // for (int i = 0; i < row_order.size(); i++){
    // 	row_order2(i) = row_order;
    // }
    
    // Eigen::VectorXi row_order_vec(Eigen:Dynamic);
    // row_order_vec << row_order;
    // std::cout << row_order_vec;
    // std::cout << row_order;

    // Eigen::VectorXi row_order2;

    // int i = 0;
    
    // for (auto item : row_order) {
    //     std::cout << item << "\n";
    // 	row_order2(i) = item;
    // 	i++;
    
    // }
    
    // Eigen::VectorXi row_order(4);
    // row_order << 1,2,3,4;
    
    std::cout << row_order;

    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> perm;
    perm.indices() = row_order;

    return perm;
}

void blocker2(int SIZE, Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> perm) {

  Eigen::MatrixXf mat = Eigen::MatrixXf::Random(SIZE * 2, SIZE * 2);

  std::cout << mat << std::endl << std::endl;

  mat.resize((SIZE * SIZE * 2), 2);
  
  std::cout << mat << std::endl << std::endl;
  std::cout << perm * mat << std::endl << std::endl;;
 
}


void timer_perm (int SIZE, Eigen::VectorXi row_order) {
    // get perm
    
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> perm;
    perm = create_perm(row_order);
    // std::cout << perm.toDenseMatrix();

    // run with same perm matrix multiple times
    for (int i=0; i<2; i++){
    	blocker2(SIZE, perm);
    }
}

PYBIND11_MODULE(eigen_block_test, m) {
    m.doc() = "block test";
    m.def("blocker1", &blocker1, "asdf");
    m.def("blocker3", &blocker3, "asdf");
    // m.def("blocker2", &blocker2, "asdf");
    // m.def("create_perm", &create_perm, "asdf");
    m.def("timer_perm", &timer_perm, "asdf");
}

