#include <iostream>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <Eigen/unsupported/Eigen/CXX11/Tensor>

// #include <list>

/** 
    get matrix blocks with eigen blocks, unoptimized

 */
void blocker1 (int SIZE, Eigen::MatrixXf mat) {

    // Eigen::MatrixXf mat(4,4);
    // mat << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;

    // Eigen::MatrixXf mat = Eigen::MatrixXf::Random(SIZE*2,SIZE*2);
    Eigen::MatrixXf res_mat(SIZE,SIZE);
    
    // #pragma omp parallel for
    float cell = 0;
    for (int i=0; i<SIZE; i++){
	for (int k=0; k<SIZE-i; k++) {
	    cell = mat.block<2,2>(i*2,k*2).maxCoeff();

	    res_mat(i,k) = cell;
	    // res_mat(k,i) = cell;
	}
    }
    // std::cout << mat.block<2,2>(0,0);

    
    // std::cout << "res_mat block: \n" << res_mat << "\n";

    
    // return res_mat;
}



/**
   check how to get block-reduced array without looping over blocks
*/
void blocker3 () {


    Eigen::MatrixXi M1(4,4);    // Column-major storage
    M1 << 1, 2, 3,
	4, 5, 6,
	7, 8, 9, 10, 11,12,13,14,15,16;

    // std::cout << "M1" << std::endl << M1 << std::endl << std::endl;


    Eigen::Map<Eigen::RowVectorXi> v1(M1.data(), M1.size());
    // std::cout << "v1:" << std::endl << v1 << std::endl << std::endl;
    
    Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> M2(M1);

    // std::cout << "M2:" << std::endl << M2 << std::endl << std::endl;

    
    
    // Eigen::Map<Eigen::RowVectorXf> v2(M2.data(), M2.size());

    // Eigen::Map<Eigen::MatrixXi> v2(M2.data(), 8, 2);
    Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> v2(M1.data(), 8, 2);


    // Eigen::Map<Eigen::MatrixXf<Eigen::Dynamic,Eigen::Dynamic> > v2(M2.data());
    // Eigen::Map<Eigen::Matrix<int, 8, 2, Eigen::RowMajor> > v2(M2.data());
    // std::cout << "v2:" << std::endl << v2 << std::endl;
    // std::cout << Eigen::Map < Eigen::Matrix<int,8,2> > (M1);

    // int array[8];
    // for(int i = 0; i < 8; ++i) array[i] = i;
    // std::cout << "Row-major:\n" << Eigen::Map<Eigen::Matrix<int,4,2, Eigen::RowMajor> >(array) << std::endl;

    // std::cout << "next try :\n" << Eigen::Map<Eigen::Matrix<int,8,2, Eigen::RowMajor> >(M2.data()) << std::endl;

    
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
    
    // std::cout << row_order;

    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> perm;
    perm.indices() = row_order;

    
    return perm.transpose();
}

/** 
    get block information through reordering matrix

 */


void blocker2(Eigen::MatrixXf mat, int SIZE, Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> perm) {

    // Eigen::MatrixXf mat = Eigen::MatrixXf::Random(SIZE * 2, SIZE * 2);

    // std::cout << "input dist mat: \n" << mat << std::endl << std::endl;

    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> mat_rowmaj(mat);

    
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_rszd(mat_rowmaj.data(), SIZE*SIZE*2, 2);

      
    // std::cout << "mat reshaped: \n" << mat_rszd << std::endl << std::endl;
    // std::cout << perm.toDenseMatrix() << std::endl << std::endl;
    /// std::cout << perm * mat_rszd << std::endl << std::endl;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> res_mat(SIZE*SIZE*2, 2);

    

    res_mat << perm * mat_rszd;
    // std::cout << "rows reorderd: \n" << res_mat;
    

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> res_rord(res_mat.data(), SIZE*SIZE, 4);

    // std::cout << "reorderd res: \n" << res_rord << std::endl;

    
    Eigen::VectorXf max_vec(SIZE*SIZE);
    
    max_vec = res_rord.rowwise().maxCoeff();

    // std::cout <<"maxvec: \n" << max_vec << "\n";

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> max_mat(max_vec.data(), SIZE, SIZE);

    // std::cout <<"max mat: \n" << max_mat << "\n";
    
    //std::cout << "max coefs: \n" << res_rord.rowwise().maxCoeff();
    
}

/** 
    see if mapping can theoretically be fast enough 
    results incorrect for now, just technical implementation
 */

void perm_basic (Eigen::MatrixXf mat, int SIZE, Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> perm) {
    
    Eigen::Map<Eigen::MatrixXf> mat_rszd(mat.data(), SIZE*SIZE*2, 2);

    Eigen::MatrixXf res_mat(SIZE*SIZE*2, 2);
    res_mat << perm * mat_rszd;
    
    Eigen::Map<Eigen::MatrixXf> res_rord(res_mat.data(), SIZE*SIZE, 4);
    // Eigen::Map<Eigen::MatrixXf> res_rord(mat.data(), SIZE*SIZE, 4); // for skipping permutation
    

    Eigen::VectorXf max_vec(SIZE*SIZE);
    max_vec = res_rord.rowwise().maxCoeff();

    // Eigen::Map<Eigen::MatrixXf> max_mat(max_vec.data(), SIZE, SIZE);
	
}




void tensor_func2 (int SIZE, Eigen::MatrixXf stor_mat) {
    // int array[SIZE*SIZE*4];
    // for(int i = 0; i < SIZE*SIZE*4; ++i) array[i] = i;
    // std::cout << "array" << array;

    // Eigen::MatrixXi stor_mat = Eigen::MatrixXi::Random(SIZE*2, SIZE*2);
        
    // Eigen::MatrixXd m = (Eigen::MatrixXd::Random(SIZE*2,SIZE*2)+Eigen::MatrixXd::Ones(SIZE*2,SIZE*2))*5;  
    // Eigen::MatrixXi stor_mat = m.cast<int>();

    // std::cout << "stor_mat: \n" << stor_mat;

    Eigen::Map<Eigen::MatrixXf, 0, Eigen::InnerStride<2>> stride1 (stor_mat.data(), SIZE,SIZE*2);
    Eigen::Map<Eigen::MatrixXf, 0, Eigen::InnerStride<2>> stride2 (stor_mat.bottomRows(2*SIZE-1).data(), SIZE,SIZE*2);

    // std::cout << "\nstride1: \n" << stride1;
    // std::cout << "\nstride2: \n" << stride2;

    Eigen::MatrixXf join_mat(SIZE*2, SIZE*2);
    join_mat << stride1, stride2;

    // std::cout << "\njoin_mat: \n" << join_mat;

    Eigen::TensorMap<Eigen::Tensor<float, 3>> tt(join_mat.data(), SIZE,4,SIZE);
    
    Eigen::array<int, 1> dims({1}); //dimensions to reduce

    Eigen::Tensor<float, 2> b = tt.maximum(dims);
    // std::cout << "\nmax values: \n" << b;

}


void tensor_func_no_stride (int SIZE, Eigen::MatrixXf stor_mat) {
    Eigen::TensorMap<Eigen::Tensor<float, 3>> tt(stor_mat.data(), SIZE,4,SIZE);
    Eigen::array<int, 1> dims({1}); //dimensions to reduce

    Eigen::Tensor<float, 2> b = tt.maximum(dims); 
}


void timer_perm (Eigen::MatrixXf dists, int SIZE, Eigen::VectorXi row_order) {
    // get perm
    
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> perm;
    perm = create_perm(row_order);
    // std::cout << perm.toDenseMatrix();

    // run with same perm matrix multiple times
    for (int i=0; i<250; i++){
    	blocker2(dists, SIZE, perm);
    }
}

/** 
    just map a lot and see how expensive it gets

*/

void map_tester (Eigen::MatrixXf stor_mat, int nbr_map){

    for (int i=0; i < nbr_map; i++){
	
	Eigen::Map<Eigen::MatrixXf> stride1 (stor_mat.data(), stor_mat.rows()/2 ,stor_mat.cols()*2);
    }
}
    

void timer_perm_basic (Eigen::MatrixXf dists, int SIZE, Eigen::VectorXi row_order) {
    // get perm
    
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> perm;
    perm = create_perm(row_order);
    // std::cout << perm.toDenseMatrix();

    // run with same perm matrix multiple times
    for (int i=0; i<250; i++){
    	perm_basic(dists, SIZE, perm);
    }
}


void timer_block (int SIZE, Eigen::MatrixXf dists) {
    for (int i=0; i<250; i++){
	blocker1(SIZE, dists);
    }
}
    

void timer_tensor (int SIZE, Eigen::MatrixXf dists) {
    for (int i=0; i<250; i++){
	tensor_func2(SIZE, dists);
    }
}

void timer_tensor_no_stride (int SIZE, Eigen::MatrixXf dists) {
    for (int i=0; i<250; i++){
	tensor_func_no_stride(SIZE, dists);
    }
}


void timer_map_tester (Eigen::MatrixXf dists, int nbr_map) {
    for (int i = 0; i<250; i++){
	map_tester(dists, nbr_map);
    }
}

PYBIND11_MODULE(eigen_block_test, m) {
    m.doc() = "block test";
    m.def("blocker1", &blocker1, "asdf");
    m.def("blocker3", &blocker3, "asdf");
    // m.def("blocker2", &blocker2, "asdf");
    // m.def("create_perm", &create_perm, "asdf");
    m.def("timer_perm", &timer_perm, "asdf");
    m.def("timer_perm_basic", &timer_perm_basic, "asdf");
    m.def("timer_block", &timer_block, "asdf");
    m.def("timer_tensor", &timer_tensor, "asdf");
    m.def("timer_tensor_no_stride", &timer_tensor_no_stride, "asdf");
    m.def("timer_map_tester", &timer_map_tester, "asf");
}
