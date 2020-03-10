#include <iostream>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// something to get indices of matrix where condition is fulfilled
// https://stackoverflow.com/questions/50027494/eigen-indices-of-dense-matrix-meeting-condition
template<typename Func>
struct lambda_as_visitor_wrapper : Func {
    lambda_as_visitor_wrapper(const Func& f) : Func(f) {}
    template<typename S,typename I>
    void init(const S& v, I i, I j) { return Func::operator()(v,i,j); }
};

template<typename Mat, typename Func>
void visit_lambda(const Mat& m, const Func& f)
{
    lambda_as_visitor_wrapper<Func> visitor(f);
    m.visit(visitor);
}    


/**
   calculate dists
   first get center dists, then refine those where distances are under some threshold involving node dimensions

 */


// how to do delta nodes




/**
   calculate 1d deltas
   can probably be used to construct deltas overall
 */

Eigen::MatrixXf dists1d(Eigen::VectorXf v) {
    int SIZE = v.rows();

    Eigen::MatrixXf mat(SIZE, SIZE);
    mat = v.transpose().replicate(SIZE, 1).colwise() - v;
    return mat;
}

/** 
    calculate pairwise distance matrix
 */

Eigen::MatrixXf dists(Eigen::MatrixXf pts_pos){
    // D = ( (p0.transpose() * p1 * -2
    //   ).colwise() + p0.colwise().squaredNorm().transpose()
    // ).rowwise() + p1.colwise().squaredNorm();


    Eigen::MatrixXf D(pts_pos.rows(),pts_pos.rows());
    /** some long comment 

    std::cout << "\ncoords: \n" << pts_pos;
    std::cout << "\ntransposed: " << pts_pos.transpose();
    Eigen::VectorXf vec_par1;
    vec_par1 = pts_pos.rowwise().squaredNorm().transpose();
    std::cout << "\nsome vector part: " << vec_par1;

    some modified stuff that seems to work
    https://stackoverflow.com/questions/19280782/pairwise-differences-between-two-matrices-in-eigen
    */
    

    D = ((pts_pos * pts_pos.transpose() *  -2
	).colwise() + pts_pos.rowwise().squaredNorm()
	).rowwise() + pts_pos.rowwise().squaredNorm().transpose();

    // std::cout << "\ndist mat in c: \n" << D.array().sqrt();
    // return D.array().sqrt();
    return D.cwiseSqrt();
}

/** 
    check which distances need their borders checked
 */

std::vector<std::pair<int,int>> max_dim_ext(Eigen::MatrixXf dists, Eigen::MatrixXf dim_ar, int SIZE) {
    Eigen::VectorXf max_dim(SIZE);
    
    max_dim = dim_ar.rowwise().maxCoeff();

    // std::cout << "\nmax_dim: " << max_dim;

    Eigen::MatrixXf max_dim_dists(SIZE, SIZE);

    // std::cout << "\ntransposed: \n" << max_dim.transpose();
    
    max_dim_dists = max_dim.transpose().replicate(SIZE, 1).colwise() + max_dim;

    max_dim_dists.diagonal().setZero();
    // std::cout << "\nmax_dim_dists:\n" << max_dim_dists;

    // subtract dims
    Eigen::MatrixXf diff_dists(SIZE,SIZE);
    diff_dists.triangularView<Eigen::Lower>() = dists - 2*max_dim_dists;

    std::cout << "\ndiff_dists: \n" << diff_dists;

    int th = 0;

    std::vector<std::pair<int,int>> indices;

    visit_lambda(diff_dists,
        [&indices,th](double v, int i, int j) {
            if(v < th)
                indices.push_back(std::make_pair(i,j));
        });
    
    
    // std::vector<std::pair<int,int>> indices2;
    
    // visit_lambda(diff_dists,
    //     [&indices2,th](double v, int i, int j) {
    //         if(th == th)
    //             indices2.push_back(std::make_pair(1,1));
    //     });

    // int ctr = 0;

    // for(auto p:indices2)
    // 	ctr++;
    // std::cout << "\nctr: " << ctr;
        
	
    // for(auto p:indices)
    //     std::cout << '(' << p.first << ',' << p.second << ") ";
    // std::cout << '\
	n';

    return indices;
    
    
}
    



PYBIND11_MODULE(frucht_v3, m) {
  m.doc() = "aasdf";
  m.def("dists", &dists, "asdf");
  m.def("max_dim_ext", &max_dim_ext, "asdf");
}
