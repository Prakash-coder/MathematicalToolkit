#include <iostream>
//#include <string>
#include <chrono>
#include <valarray>
#include <algorithm>
#include "../src/MathematicalToolkit.hpp"
//using namespace linalg;
//using linalg::NDual;
using linalg::Vector;
using linalg::NDual;
//using linalg::Vector;

//using namespace std::chrono;

const double eps = 10e-5;

template<typename T>
T f(const T& x, const T& y)
{
	return (x + y)*5.;
}

template <typename T>
struct function { T operator()( const T& x, const T& y){ return (x+y)*5.; } };

template<typename T, typename F>
Vector<T> f_finite_differencing( F f, const T& x, const T& y )
{
	Vector<T> result ({ (f(x+eps, y)-f(x,y))/eps, (f(x,y+eps)- f(x,y))/eps });
	return result;
}

inline double inner_product( Vector<double>& vector1, Vector<double>& vector2, double& return_val )
{
	return_val = vector1 * vector2;
	return return_val;
}

template <typename T>
void print( const std::valarray<T>& arr1 )
{
	for ( auto i = std::begin(arr1); i != std::end(arr1); ++i )
	{
		std::cout << *i << std::endl;
	}
}




int main()
{
	//Vector<double> a ({1,0});
	//Vector<double> b ({0,1});

	//NDual <double> x    ( 1, a );
	//NDual <double> y    ( 1, b );
	//NDual<double> result = f<NDual<double>>( x, y );
	//result.Show();
	//std::valarray<double> c = a*5;
	//print<double>( std::sin(a) );
	//double x = 1;
	//double y = 1;
	//Vector<double> gradient = f_finite_differencing<>( function<double>(), x, y );
	//gradient.Show();
	//std::cout << "\n" << std::endl;

	Vector<double> vec1, vec2, vec3;
	double return_val{0};


	//Vector<double> vec4({0,1,2,3,4,5});
	//Vector<double> vec5  = vec4*5.;
	//vec5.Show();


	auto start = std::chrono::high_resolution_clock::now();
	for ( auto i = 0; i < 100000; ++i )
	{
		vec1.push_back(rand());
		vec2.push_back(rand());
	}

	inner_product( vec1, vec2, return_val );
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast< std::chrono::microseconds >( stop - start );
	std::cout << " the time to fill 100000 elements followed by multiply is: " << duration.count() << std::endl;
	std::cout << return_val << std::endl;
}
	
	//std::cout << std::boolalpha << linalg::dual_impl::can_become_dual_v<char> << std::endl;
	//return 0;
//}

	//NDual <double> x    ( 1, a );
	//NDual <double> y    ( 1, b );
	//NDual<double> result = sqnorm<>( x, y );
     //result.Show();
	 //Tensor<double, 1> vec2 = {0,1,2};
	 //Tensor<double, 1> vec3 = (vec1 + vec2);
	//std::cout << ve3 << std::endl;



