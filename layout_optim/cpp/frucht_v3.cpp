#include <iostream>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <chrono>




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
    // return mat.transpose();
    return (-1*mat.array());
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

    // std::cout << "\nmin D: " << D.minCoeff();
    
    return D.cwiseAbs().cwiseSqrt();
}

/** 
    check which distances need their borders checked
    takes:
    - distance matrix
    - max dimension distance ?? 
    - nbr nodes (size) to avoid having to calculate it again
*/

std::vector<std::pair<int,int>> max_dim_ext(const Eigen::Ref<const Eigen::MatrixXf> dists,
					    const Eigen::Ref<const Eigen::MatrixXf> max_dim_dists,
					    const int SIZE) {

    // subtract dims
    Eigen::MatrixXf diff_dists(SIZE,SIZE);
    diff_dists.triangularView<Eigen::Lower>() = dists - max_dim_dists;
    diff_dists.triangularView<Eigen::Upper>() = Eigen::MatrixXf::Constant(SIZE, SIZE,1);

    // std::cout << "\ndiff_dists: \n" << diff_dists;

    // int th = 0;

    std::vector<std::pair<int,int>> indices;

    visit_lambda(diff_dists,
		 // [&indices,th](double v, int i, int j) {
		 // if(v < th)
		 [&indices](double v, int i, int j) {
		     if(std::signbit(v))
			 indices.push_back(std::make_pair(i,j));
		 });

    
    

    // std::cout << "\nindices:";
    // for(auto p:indices)
    //     std::cout << '(' << p.first << ',' << p.second << ") ";
    // std::cout << '\n';

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
    D = ( (p1 * p0.transpose() * -2
	      ).colwise() + p1.rowwise().squaredNorm()
	).rowwise() + p0.rowwise().squaredNorm().transpose();

    // D = ( (p1 * p0.transpose() * -2
    //       ).colwise() + p0.colwise().squaredNorm().transpose()
    // ).rowwise() + p1.colwise().squaredNorm();

    // D = ((pts_pos * pts_pos.transpose() *  -2
    // 	     ).colwise() + pts_pos.rowwise().squaredNorm()
    // 	).rowwise() + pts_pos.rowwise().squaredNorm().transpose();
    // std::cout << "\ndists:\n" << D;

    // debug: 
    // if (isnan(std::sqrt(D.minCoeff()))) {
    // 	std::cout << "\nD:\n" << D;
    // 	std::cout << "\np0:\n" << p0;
    // 	std::cout << "\np1:\n" << p1;
    // }

    return std::sqrt((D.cwiseAbs()).minCoeff());

    // asdf
}

/** 
    updates the dist where the nodes are close together

*/

void update_dists(const std::vector<std::pair<int, int>> indices,
                  Eigen::Ref<Eigen::MatrixXf> dists_nd,
		  const Eigen::Ref<const Eigen::MatrixXf> dim_ar,
		  const Eigen::Ref<const Eigen::MatrixXf> pos_nds,
		  int *both_ovlp_count) {

    
    // std::cout << "\npos_nds:\n" << pos_nds;

    // get unique indices and put them to map
    std::vector<int> unique_indices;
    unique_indices = getElements(indices);
    // for (auto i : unique_indices) {
    // 	std::cout << "\nunique indices: " << i;
    // }

    std::map<int,Eigen::Matrix<float,4,2>> corner_map;

    for (auto i : unique_indices) {
	Eigen::Matrix<float,4,2> corner_mat;
	corner_mat = get_corner_points(pos_nds(i, 0), pos_nds(i, 1),dim_ar(i, 0), dim_ar(i, 1));
	// std::cout << "\ncorner points:\n" << corner_mat;
	corner_map[i] = corner_mat;
    }
    
    // std::cout << "\nmap array length: " << corner_map.size();

    // timer
    // auto start = std::chrono::steady_clock::now();
    
    // openmp: would only result in improvement for bigger graphs (100s of nodes)
    // #pragma omp parallel for
    // for (int itr=0; itr < indices.size(); itr++) {
    // auto p = indices[itr];
    
    for (auto p : indices) {
	// std::cout << "\n(" << p.first << ',' << p.second << ") ";
	// get node extremes in x
	float p1_left = pos_nds(p.first,0) - dim_ar(p.first,0),p1_right = pos_nds(p.first,0) + dim_ar(p.first,0);
	float p2_left = pos_nds(p.second,0) - dim_ar(p.second,0),p2_right = pos_nds(p.second,0) + dim_ar(p.second,0);
	// get node extremes in y
	float p1_bottom = pos_nds(p.first,1) - dim_ar(p.first,1),p1_top = pos_nds(p.first,1) + dim_ar(p.first,1);
	float p2_bottom = pos_nds(p.second,1) - dim_ar(p.second,1),p2_top = pos_nds(p.second,1) + dim_ar(p.second,1);

	Eigen::Vector4f xdists, ydists;
	xdists << p2_left - p1_left, p2_left - p1_right,  p2_right - p1_left,  p2_right - p1_right;
	ydists << p2_bottom - p1_bottom, p2_bottom - p1_top,  p2_top - p1_bottom,  p2_top - p1_top;

	// signbit: product is negative (signbit =1) if overlap exists
	float x_ovlp = std::signbit(xdists.minCoeff() * xdists.maxCoeff());
	float y_ovlp = std::signbit(ydists.minCoeff() * ydists.maxCoeff());
	
	float x_shrt = xdists.array().abs().minCoeff();
	float y_shrt = ydists.array().abs().minCoeff();
	
	// get min corner_point_distances
	float min_corner_dist;
	min_corner_dist = get_min_corner_dist(corner_map[p.first], corner_map[p.second]);
	// std::cout << "\nmin corner dist: " << min_corner_dist;
	// std::cout << "\nold dist: " << dists_nd(p.first, p.second);
	
	float both_ovlp = x_ovlp * y_ovlp;
	float x_ovlp2 = x_ovlp - both_ovlp;
	float y_ovlp2 = y_ovlp - both_ovlp;
	float none_ovlp = 1 - both_ovlp - x_ovlp2 - y_ovlp2;
	
	// std::cout << "\nx_ovlp2: " << x_ovlp2;
	// std::cout << "\ny_ovlp2: " << y_ovlp2;
	// std::cout << "\nboth_ovlp2: " << both_ovlp;
	// std::cout << "\nnone_ovlp2: " << none_ovlp;


	float new_dist = (x_ovlp2 * y_shrt) + (y_ovlp2 * x_shrt) + both_ovlp + (none_ovlp * min_corner_dist);
        if (isnan(new_dist)) {
	    std::cout << "\nNEW DIST IS NAN\n";
	    std::cout << "\nx_ovlp2, y_shrt: " << x_ovlp2 << " " << y_shrt;
	    std::cout << "\ny_ovlp2, x_hsrt: " << y_ovlp2 << " " << x_shrt;
	    std::cout << "\nboth ovlp: " << both_ovlp;
	    std::cout << "\nnone_ovlp, min_corner_dist: " << none_ovlp << " " << min_corner_dist;
	    new_dist = 1;
        }
        dists_nd(p.first, p.second) = new_dist;
	dists_nd(p.second, p.first) = new_dist;

	
	// update pointer with overlap counter
	*both_ovlp_count += both_ovlp;
	// std::cout << both_ovlp;
	// std::cout << "\nnew dist: " << new_dist;
    }
        
    // auto end = std::chrono::steady_clock::now();
    
    // std::cout << "\nElapsed time in microseconds : " 
    // 	      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
    // 	      << " µs" << std::endl;

    // do stuff
}

     
/**
   entire function
   pos_nd: node position
   don't need no corner matrix
   k: something about ideal distance
   A: connectivity matrix
   width and height
   t: temperature 
   max_itr: max amount of iterations
   def_itr: minimum number of iterations
   rep_nd_brd_start: percentage of last section of def_itr when to start making node borders repellent
*/

Eigen::MatrixXf frucht(Eigen::MatrixXf pos_nd, Eigen::MatrixXf dim_ar, float k, Eigen::MatrixXf A,
		       float width, float height, float t, int max_itr, int def_itr, float rep_nd_brd_start) {
    // setup objects
    int NBR_NDS = pos_nd.rows();
    

    // setup objects done
   
    int ctr = 0;
    float dt = t/def_itr;
    int both_ovlp_count = 0;
    float boundaries_crossed = 0;

    // std::cout << "\nconstant vector test:\n"  << Eigen::VectorXf::Constant(10,1);
    // std::cout << "\nnode positions:\n" << pos_nd;

    // generate max dim array (doesn't change):
    // indicates how far nodes can be apart through their extent
    // if actual distance is smaller than this, nodes get border calculations
    
    Eigen::VectorXf max_dim(NBR_NDS);
    max_dim = (dim_ar.rowwise().maxCoeff()).array()*2;
    Eigen::MatrixXf max_dim_dists(NBR_NDS, NBR_NDS);
    max_dim_dists = max_dim.transpose().replicate(NBR_NDS, 1).colwise() + max_dim;
    max_dim_dists.diagonal() = Eigen::VectorXf::Constant(NBR_NDS,1);

    
    auto start = std::chrono::steady_clock::now();
    std::vector<std::pair<int,int>> nan_indices;

    while (true) {

	both_ovlp_count = 0;

	// visit_lambda(pos_nd,
	// 	     [&nan_indices](double v, int i, int j) {
	// 		 if(isnan(v))
	// 		     nan_indices.push_back(std::make_pair(i,j));
	// 	     });
	// if (nan_indices.size() > 0) {
	//     std::cout << "\nnode pos array:\n" << pos_nd;
	//     break;
	// }

          
	// delta calculations
	Eigen::MatrixXf delta_x, delta_y;
	delta_x = dists1d(pos_nd.col(0));
	delta_y = dists1d(pos_nd.col(1));
	// std::cout << "\ndelta_x:\n" << delta_x;
	// std::cout << "\ndelta_y:\n" << delta_y;

	Eigen::MatrixXf dists_nd = dists(pos_nd);


	// visit_lambda(dists_nd,
	// 	     [&nan_indices](double v, int i, int j) {
	// 		 if(isnan(v))
	// 		     nan_indices.push_back(std::make_pair(i,j));
	// 	     });
	// if (nan_indices.size() > 0) {
	//     std::cout << "\nnode dist array:\n" << dists_nd;
	//     break;
	// }

	std::vector<std::pair<int,int>> indices;
	indices = max_dim_ext(dists_nd, max_dim_dists, NBR_NDS);
    
	// std::cout << "\nold dists:\n" << dists_nd;
	dists_nd.diagonal() = Eigen::VectorXf::Constant(NBR_NDS,1);
    
	// point phase does not need explicit calling: just means center distances are not modified
	if (t < dt * def_itr * rep_nd_brd_start) {

	    update_dists(indices, dists_nd, (0.5 * dim_ar.array()).matrix(), pos_nd,
			 &both_ovlp_count);
	}
	

	// calculate force array
	// (k * k / distance**2) - A * distance / k
	Eigen::ArrayXXf force_ar(NBR_NDS, NBR_NDS);
	force_ar = (k*k / (dists_nd.array()).pow(2)) - ((A.array() * dists_nd.array())/k) ;
	
	// visit_lambda(force_ar,
	// 		   [&nan_indices](double v, int i, int j) {
	// 		     if(isnan(v))
	// 		       nan_indices.push_back(std::make_pair(i,j));
	// 		   });
	// if (nan_indices.size() > 0) {
	// 	std::cout << "\ndists_nd:\n:" << dists_nd;
	// 	std::cout << "\nforce array:\n" << force_ar;
	// 	std::cout << "\nindices:\n";
	//   for (auto p : indices) {
	//     std::cout << "(" << p.first << ", " << p.second << ")";
	//   }
	//   std::cout << "\npos_nd:\n" << pos_nd;
	// 	break;
	// }

	// // displacement x
	Eigen::Matrix<float,Eigen::Dynamic, 2> disp (NBR_NDS, 2);
	// std::cout << "\nsome disp:\n" << (delta_x.array() * force_ar.array()).rowwise().sum();
	disp.col(0) = (delta_x.array() * force_ar.array()).rowwise().sum();
	disp.col(1) = (delta_y.array() * force_ar.array()).rowwise().sum();
	// std::cout << "\ndisplacement:\n" << disp;
    
	// repellant window borders
	disp.col(0) = disp.col(0).array() + k*10*k*10/((pos_nd.col(0) - dim_ar.col(0)/2).array()).pow(2);
	disp.col(0) = disp.col(0).array() - k*10*k*10/(width - ((pos_nd.col(0) + dim_ar.col(0)/2).array())).pow(2);
	disp.col(1) = disp.col(1).array() + k*10*k*10/((pos_nd.col(1) - dim_ar.col(1)/2).array()).pow(2);
	disp.col(1) = disp.col(1).array() - k*10*k*10/(height - ((pos_nd.col(1) + dim_ar.col(1)/2).array())).pow(2);

	// disp = disp.cwiseMax(-1000000);
	// disp = disp.cwiseMin(1000000);

	// myarray = myarray.unaryExpr([](double v) { return std::isfinite(v)? v : 0.0; });
	// https://stackoverflow.com/questions/22927323/how-do-i-find-and-replace-all-non-finite-numbers-in-an-eigenarray-object
	disp = (disp.array().unaryExpr([](float v) { return std::isfinite(v)? v : 0; })).matrix();

	// std::cout << "\ndisp repellant borders:\n" << disp;
    
	// no gravity for now

	// delta calcs
	// standardize displacement vectors? (i guess)
	Eigen::VectorXf len_vec(NBR_NDS);
	len_vec = (disp.rowwise().norm()).cwiseMax(0.1);
	
      
	Eigen::ArrayX2f len_ar(NBR_NDS,2);
	len_ar = (t/len_vec.array()).replicate(1,2);

	Eigen::Matrix<float, Eigen::Dynamic, 2> delta_pos (NBR_NDS, 2);
	delta_pos = disp.array() * len_ar;

	// std::cout << "\ndelta_pos:\n" << delta_pos;

	// visit_lambda(delta_pos,
	// 	     [&nan_indices](double v, int i, int j) {
	// 		 if(isnan(v))
	// 		     nan_indices.push_back(std::make_pair(i,j));
	// 	     });

      
	// if (nan_indices.size() > 0) {
	//     std::cout << "\nctr: " << ctr;
	//     std::cout << "\nmin len: " << len_vec.minCoeff();
	//     std::cout << "\nlen_vec:\n" << len_vec;
	//     std::cout << "\ndisp:\n" << disp;
	//     std::cout << "\nlen_ar:\n" << len_ar;
	//     std::cout << "\ndelta pos:\n" << delta_pos;

	//     pos_nd(0,0) = 0/0;
	//     pos_nd(0,1) = std::sqrt(-2);
	//     std::cout << "\npos nodes:\n" << pos_nd;
	//     break;
	// }

	// update logic
	pos_nd += delta_pos;

      
      
	// check boundary crossings

	boundaries_crossed = 0;
	if (pos_nd.col(0).minCoeff() < 0 or pos_nd.col(0).maxCoeff() > width or
	    pos_nd.col(1).minCoeff() < 0 or pos_nd.col(1).maxCoeff() > height) {
	    boundaries_crossed = 1;
	}
      
	// std::cout << "\nctr: " << ctr << "; t: " << t << "; both_ovlp_count: " << both_ovlp_count;

    
	ctr ++;
	if (ctr == max_itr or t < 0) {break;}
    
	// reduce temperature in first phase (no node borders)
	if (t > (dt * def_itr * rep_nd_brd_start)) {
	    t -= dt;
	}

	// reduce in second phase if no ovlp and no boundary crossing
	else {
	    if (both_ovlp_count == 0 and boundaries_crossed == 0) {
		t -= dt;
	    }
	}
    }

    auto end = std::chrono::steady_clock::now();
    
    // std::cout << "\nElapsed time in microseconds : " 
    // 	      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
    // 	      << " µs" << std::endl;

    // std::cout << "\nnode pos:\n" << pos_nd;
    return pos_nd;
}

PYBIND11_MODULE(frucht_v3, m) {
    m.doc() = "aasdf";
    m.def("dists", &dists, "asdf");
    m.def("max_dim_ext", &max_dim_ext, "asdf");
    m.def("update_dists", &update_dists, "asdf");
    m.def("get_corner_points", &get_corner_points, "asdf");
    m.def("get_min_corner_dist", &get_min_corner_dist, "asdf");
    m.def("frucht", &frucht, "asdf");
}
