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


// how to do delta nodes


/**
   calculate 1d deltas
   can probably be used to construct deltas overall
 */

Eigen::MatrixXf dists1d(const Eigen::Ref<const Eigen::VectorXf> v) {
    int SIZE = v.rows();

    Eigen::MatrixXf mat(SIZE, SIZE);
    mat = v.transpose().replicate(SIZE, 1).colwise() - v;
    return mat;
}

/** 
    calculate pairwise distance matrix
 */

Eigen::MatrixXf dists(const Eigen::Ref<const Eigen::MatrixXf> pts_pos){
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
    takes:
    - distance matrix
    - node dimension array
    - nbr nodes (size) to avoid having to calculate it again
 */

std::vector<std::pair<int,int>> max_dim_ext(const Eigen::Ref<const Eigen::MatrixXf> dists,
					    const Eigen::Ref<const Eigen::MatrixXf> dim_ar,
					    const int SIZE) {
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

/** 
    get unique nodes from pair vector
*/

std::vector<int> getElements(std::vector<std::pair<int, int>> S) {
    std::unordered_set<int> ss;
    std::for_each(S.begin(), S.end(), [&ss](const auto& p) {
        ss.insert(p.first);
        ss.insert(p.second);
    });
    return std::vector<int>(ss.begin(), ss.end());
}


/** 
    construct corner points
    not really possible to construct them only once since they change with nodes changing
    hopefully using the specified size types is faster? 
    use Matrix4Xf or MatrixXf<4,2>?
*/

Eigen::Matrix<float,4,2> get_corner_points(float x, float y,
					   float width, float height) {
    Eigen::Matrix<float,4, 2> corner_mat;
    corner_mat(0, 0) = x - width;
    corner_mat(0, 1) = y + height;
    corner_mat(1, 0) = x + width;
    corner_mat(1, 1) = y + height;
    corner_mat(2, 0) = x + width;
    corner_mat(2, 1) = y - height;
    corner_mat(3, 0) = x - width;
    corner_mat(3, 1) = y - height;
  
  return corner_mat;
}

/** 
    calculate the 4x4 point distances (previously blocks)
 */

float get_min_corner_dist(Eigen::Ref<Eigen::Matrix<float, 4, 2>> p0,
			  Eigen::Ref<Eigen::Matrix<float, 4, 2>> p1) {

    // D = ((pts_pos * pts_pos.transpose() *  -2
    // 	     ).colwise() + pts_pos.rowwise().squaredNorm()
    // 	).rowwise() + pts_pos.rowwise().squaredNorm().transpose();



    Eigen::Matrix<float,4,4> D;
    D = ( (p0 * p1.transpose() * -2
	      ).colwise() + p0.rowwise().squaredNorm()
	).rowwise() + p1.rowwise().squaredNorm().transpose();

    // D = ( (p1 * p0.transpose() * -2
    //       ).colwise() + p0.colwise().squaredNorm().transpose()
    // ).rowwise() + p1.colwise().squaredNorm();

    // D = ((pts_pos * pts_pos.transpose() *  -2
    // 	     ).colwise() + pts_pos.rowwise().squaredNorm()
    // 	).rowwise() + pts_pos.rowwise().squaredNorm().transpose();
    std::cout << "\ndists:\n" << D;

    return std::sqrt(D.minCoeff());

    // asdf
}

/** 
    updates the dist where the nodes are close together

 */

void update_dists(const std::vector<std::pair<int, int>> indices,
                  Eigen::Ref<Eigen::MatrixXf> dists_nd,
		  const Eigen::Ref<const Eigen::MatrixXf> dim_ar,
		  const Eigen::Ref<const Eigen::MatrixXf> pos_nds) {


    //     std::cout << p.first << p.second <<;
    // hmm how to get new distances most efficiently
    
    // still need to check all possibilities i guess
    // can't just take lowest: would mean that overlap in one dimension means total overlap
      
    // maybe vectorization? process all distance pairs at once?
    // array (element-wise) multiplication should work
    // index slicing not available yet tho (in dev branch) -> single case loop for now
    // x dists
    
    // std::vector<int> some_inds{0,1};

    // std::cout << "\nall pos:\n" << pos_nds;
    // std::cout << "\nsome pos:\n" << pos_nds(some_inds,0);


    std::cout << "\npos_nds:\n" << pos_nds;
    

    // std::cout << "\np1_left: " << p1_left;
    // std::cout << "\np1_right: " << p1_right;
    // std::cout << "\np2_left: " << p2_left;
    // std::cout << "\np2_right: " << p2_right;
    
    // for (auto k : xdists) {
    // 	std::cout << "\nsomevalue: " << k;
    // };

	
    // std::cout << "minimum element: " << *minmax.first << '\n';
    // std::cout << "maximum element: " << *minmax.second << '\n';

    
    // get unique indices
    std::vector<int> unique_indices;
    unique_indices = getElements(indices);
    for (auto i : unique_indices) {
	std::cout << "\nunique indices: " << i;
    }

    std::map<int,Eigen::Matrix<float,4,2>> corner_map;

    for (auto i : unique_indices) {
	Eigen::Matrix<float,4,2> corner_mat;
	corner_mat = get_corner_points(pos_nds(i, 0), pos_nds(i, 1),dim_ar(i, 0), dim_ar(i, 1));
	std::cout << "\ncorner points:\n" << corner_mat;
	corner_map[i] = corner_mat;
    }
    
    std::cout << "\nmap array length: " << corner_map.size();
    


    for (auto p : indices) {
	std::cout << "\n(" << p.first << ',' << p.second << ") ";
	// get node extremes in x
	float p1_left = pos_nds(p.first,0) - dim_ar(p.first,0),p1_right = pos_nds(p.first,0) + dim_ar(p.first,0);
	float p2_left = pos_nds(p.second,0) - dim_ar(p.second,0),p2_right = pos_nds(p.second,0) + dim_ar(p.second,0);
	// get node extremes in y
	float p1_bottom = pos_nds(p.first,1) - dim_ar(p.first,1),p1_top = pos_nds(p.first,1) + dim_ar(p.first,1);
	float p2_bottom = pos_nds(p.second,1) - dim_ar(p.second,1),p2_top = pos_nds(p.second,1) + dim_ar(p.second,1);

        std::vector<float> xdists { p2_left - p1_left, p2_left - p1_right,  p2_right - p1_left,  p2_right - p1_right};
	std::vector<float> ydists { p2_bottom - p1_bottom, p2_bottom - p1_top,  p2_top - p1_bottom,  p2_top - p1_top};
        
	auto minmax_x = std::minmax_element(xdists.begin(), xdists.end());
	auto minmax_y = std::minmax_element(ydists.begin(), ydists.end());

	float x_ovlp = std::signbit(*minmax_x.first * *minmax_x.second);
	float y_ovlp = std::signbit(*minmax_y.first * *minmax_y.second);
	// float mnmx_x_prod = *minmax_x.first * *minmax_x.second;
	
	// std::cout << "\nminmax prod: " << mnmx_prod;
	// signbit: product is negative (signbit =1) if overlap exists
	
	// std::cout << "\nsignbit: " << std::signbit(mnmx_prod);

	// hm how to get the corner point distances..
	// don't have separate corner point array anymore
	// could write function to construct it, like in python for all
	// then distances (similar to self-distances)
	// maybe get unique points before and calculate corner points for them -> no recalculation if same node is necessary multiple times 
	    
    }

    // do stuff
}

/**
   entire function
   pos_nd: node position
   pos_cr: corner positions: not clear if i actually need it much: only use it in some cases where i use only some elements
   
   


Eigen::MatrixXf frucht(Eigen::MatrixXf pos_nd, Eigen::MatrixXf pos_cr, Eigen::MatrixXf dim_ar) {
    // setup objects
    int NBR_NDS = pos_nd.rows();

    // setup objects done
    
    while (true) {
	// delta calculations
	Eigen::MatrixXf dists_nd = dists(pos_nd);

        if (point_phase == 1) {
	    // point phase calculations
	    // actually not really anything: should calculate dists before
        }

        if (node_border_phase == 1){
	    std::veactor<std::pair<int,int>> indices;
	    indices = max_dim_ext(dists_nd, dim_ar, NBR_NDS);

	    
	    
    
    }
}
 */


PYBIND11_MODULE(frucht_v3, m) {
  m.doc() = "aasdf";
  m.def("dists", &dists, "asdf");
  m.def("max_dim_ext", &max_dim_ext, "asdf");
  m.def("update_dists", &update_dists, "asdf");
  m.def("get_corner_points", &get_corner_points, "asdf");
  m.def("get_min_corner_dist", &get_min_corner_dist, "asdf");
}
