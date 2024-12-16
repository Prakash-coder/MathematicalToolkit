#ifndef DUALAD_H
#define DUALAD_H

//#include "matrix.hpp"
#include <cmath>
#include <limits>
#include <ctgmath>
#include <random>
#include <iostream>


#include "dualop.hpp"
#include "linalg.hpp"
#include "exceptions.hpp"
#include "utility.hpp"


namespace linalg
{

template < typename T >
class Gradient;
//---------------------------------//
//utility enum
//was supposed to be used with the _Generics() macro but 
//ended up never being used
enum t_typename
{
	TYPENAME_INT,
	TYPENAME_UNSIGNED_INT,
	TYPENAME_FLOAT,
	TYPENAME_LONG_DOUBLE,	
	TYPENAME_DOUBLE,
};
//---------------------------------//


//class NDual
template < typename T >
struct NDual
{
	using Val_Type                     = T;
	//default constructor
	NDual();
	//constructors	
	NDual( const NDual& ) 			   = default; //copy constructor
	NDual( NDual&& ) 				   = default; //move constructor
	NDual<T>& operator=(const NDual& ) = default; //copy assignment operator
	NDual<T>& operator=( NDual&& )     = default; //move assignment operator
	~NDual()                   = default; //destructor

	//another constructor
	NDual( const T& scalar, const Vector< T >& gradient );

	//utilities
	size_t npartials() 	   const { return partials.size(); }
	constexpr T value()    const { return this->primal;    }
	Vector<T> gradient()   const { return this->partials;  }

	void Show() const;

	operator float()   { return float(primal); }
	operator double()  { return double(primal); }	

//dual-scalar operations
	template < typename F >	
	constexpr NDual<T>& operator+=( const F& scalar );		//Scalar addition
	template < typename F >	
	constexpr NDual<T>& operator-=( const F& scalar );		//Scalar subtraction
	template < typename F >	
	constexpr NDual<T>& operator*=( const F& scalar );		//Scalar multiplication
	template < typename F >	
	constexpr NDual<T>& operator/=( const F& scalar );		//Scalar dual division
	template < typename F >	
	constexpr NDual<T>& operator+=( F&& scalar );		    //Scalar addition with rvalues
	template < typename F >	
	constexpr NDual<T>& operator-=( F&& scalar );		    //Scalar subtraction with rvalues
	template < typename F >	
	constexpr NDual<T>& operator*=( F&& scalar );		    //Scalar multiplication with rvalues
	template < typename F >	
	constexpr NDual<T>& operator/=( F&& scalar );		    //Scalar dual division with rvalues
//dual-dual assignment operations
	NDual<T>& operator+=( const NDual<T>& dual2 );  //dual addition
	NDual<T>& operator-=( const NDual<T>& dual2 );  //dual subtraction 
	NDual<T>& operator*=( const NDual<T>& dual2 );  //dual multiplication 
	NDual<T>& operator/=( const NDual<T>& dual2 );  //dual division 
//dual-dual operations more? 

	//NDual<T>& operator+( const NDual<T>& dual2 );	 //dual operations
	//NDual<T>& operator-( const NDual<T>& dual2 );	 //dual operations
	//NDual<T>& operator*( const NDual<T>& dual2 );	 //dual operations
	//NDual<T>& operator/( const NDual<T>& dual2 );	 //dual operations

		
//protected: ???? do we even need to make it protected? perhaps it does not even matter
	T         primal;
	Vector<T> partials;
};

template <typename T>
NDual<T>::NDual()
	:primal{0}
	,partials{}
{ 
	static_assert( dual_impl::can_become_dual_v<T>, "Type T must be an arithmetic type."
											      "can_become_dual_v<T> failure."); 
}

template <typename T>
NDual<T>::NDual( const T& scalar, const Vector< T >& gradient )
	:primal{scalar}
	,partials{gradient}
{ 
	static_assert(dual_impl::can_become_dual_v<T>, "Type T must be an arithmetic type."
											      "can_become_dual_v<T> failure."); 
}

template <typename T>
void NDual<T>::Show() const
{

	std::cout << this->primal << ", " << "{";
	this-> partials.Show();
	std::cout << "} eps" << std::endl;
}


//operatoroverloading in class, assignment operators
template < typename T >
template <typename F>
inline 
constexpr NDual<T>& NDual<T>::operator+=( const F& scalar ) 
{	
	static_assert(dual_impl::can_become_dual_v<F>, "Type F must be an arithmetic type."
											      "can_become_dual_v<F> failure."); 
	primal += scalar; 
	return *this; 
}

template < typename T >
template <typename F>
inline
constexpr NDual<T>& NDual<T>::operator-=( const F& scalar ) 
{
	static_assert(dual_impl::can_become_dual_v<F>, "Type F must be an arithmetic type."
											      "can_become_dual_v<F> failure."); 
	primal -= scalar; 
	return *this; 
}

template < typename T >
template <typename F>
inline
constexpr NDual<T>& NDual<T>::operator*=( const F& scalar ) 
{
	static_assert(dual_impl::can_become_dual_v<F>, "Type F must be an arithmetic type."
											      "can_become_dual_v<F> failure."); 
	primal   *= scalar; 
	partials *= scalar;
	return *this; 
}

template < typename T >
template <typename F>
inline
constexpr NDual<T>& NDual<T>::operator/=( const F& scalar ) 
{
	static_assert(dual_impl::can_become_dual_v<F>, "Type F must be an arithmetic type."
											      "can_become_dual_v<F> failure."); 
	primal   /= scalar; 
	partials /= scalar;
	return *this; 
}

template < typename T >
template <typename F>
inline
constexpr NDual<T>& NDual<T>::operator+=( F&& scalar ) 
{
	static_assert(dual_impl::can_become_dual_v<F>, "Type F must be an arithmetic type."
											      "can_become_dual_v<F> failure."); 
	primal += scalar; 
	return *this; 
}//Scalar addition

template < typename T >
template <typename F>
inline
constexpr NDual<T>& NDual<T>::operator-=( F&& scalar ) 
{
	static_assert(dual_impl::can_become_dual_v<F>, "Type F must be an arithmetic type."
											      "can_become_dual_v<T> failure."); 
	primal -= scalar; 
	return *this; 
} //Scalar subtraction

template <typename T> 
template <typename F>
inline
constexpr NDual<T>& NDual<T>::operator*=( F&& scalar ) 
{
	static_assert(dual_impl::can_become_dual_v<F>, "Type F must be an arithmetic type."
											      "can_become_dual_v<F> failure."); 
	primal   *= scalar; 
	partials *= scalar;
	return *this; 
} //Scalar multiplication

template <typename T>
template <typename F>
inline
constexpr NDual<T>& NDual<T>::operator/=( F&& scalar ) 
{
	static_assert(dual_impl::can_become_dual_v<F>, "Type F must be an arithmetic type."
											      "can_become_dual_v<F> failure."); 
	primal   /= scalar; 
	partials /= scalar;
	return *this; 
} //Scalar dual division

//binary operator overloading
template <typename T>
constexpr NDual<T> operator+( const NDual<T>& dual, const T& scalar ){ return NDual<T>( dual.primal + scalar, dual.partials ); }

template <typename T>
constexpr NDual<T> operator+( const T& scalar, NDual<T>& dual ){ return dual + scalar;	}

//template <typename T>
//constexpr NDual<T> operator*( const NDual<T>& dual, bool b ) 
//{ 
	//return b ? dual
			 //: (dual.primal > 0) ? NDual<T>(0, zeros(dual.partials)) 
								  //: NDual<T>(0, -zeros(dual.partials)); 
//}


//template <typename T>
//constexpr NDual<T> operator*( bool b, const NDual<T>& dual) { return dual * b; }
//-operator
template <typename T>
constexpr NDual<T> operator-( const NDual<T>& dual, const T& scalar ){ return NDual<T>( dual.primal - scalar, dual.partials ); }
//unary - operator need to overload scalar*vector for Vector<T>
template <typename T>
constexpr NDual<T> operator-( const NDual<T>& dual ){ return NDual<T>( -dual.primal, -dual.partials ); }

template <typename T>
constexpr NDual<T> operator-( const T& scalar, NDual<T>& dual ){ return -(dual - scalar); }

template <typename T>
constexpr NDual<T> operator*( const NDual<T>& dual, const T& scalar ){ return NDual<T>( dual.primal * scalar, dual.partials ); }

template <typename T>
constexpr NDual<T> operator*( const T& scalar, const NDual<T>& dual ){ return dual * scalar; }

template <typename T>
constexpr NDual<T> operator/( const NDual<T>& dual, const T& scalar ){ return NDual<T>( dual.primal / scalar, dual.partials ) ;}

// scalar / dual 
// //needs extensive checking
template <typename T>
constexpr NDual<T> operator/( const T& scalar, const NDual<T>& dual ) { return ( scalar/dual.primal, -std::pow( dual, -2 )*scalar ); }

template <typename T>
bool operator==( const NDual<T>& dual1, const NDual<T>& dual2 ) { return dual1.primal == dual2.primal; }

template <typename T>
bool operator<( const NDual<T>& dual1, const NDual<T>& dual2 ) { return dual1.primal < dual2.primal; }

template <typename T>
bool operator>( const NDual<T>& dual1, const NDual<T>& dual2 ) { return dual1.primal > dual2.primal; }

template <typename T>
bool operator!=( const NDual<T>& dual1, const NDual<T>& dual2 ) { return dual1.primal != dual2.primal; }

template <typename T>
bool operator<=( const NDual<T>& dual1, const NDual<T>& dual2 ) { return dual1.primal <= dual2.primal; }

template <typename T>
bool operator>=( const NDual<T>& dual1, const NDual<T>& dual2 ) { return dual1.primal >= dual2.primal; }

template <typename T>
Enable_if<dual_impl::can_become_dual_v<T> , NDual<T> > 
operator+(const NDual<T>& dual1, const NDual<T>& dual2)
{
	return ( dual1.primal + dual2.primal, dual1.partials + dual2.partials );
}


template <typename T>
Enable_if<dual_impl::can_become_dual_v<T>, NDual<T> > 
operator-(const NDual<T>& dual1, const NDual<T>& dual2)
{
	return ( dual1.primal - dual2.primal, dual1.partials - dual2.partials );
}

template <typename T>
Enable_if<dual_impl::can_become_dual_v<T>, NDual<T> > 
operator*(const NDual<T>& dual1, const NDual<T>& dual2)
{
	return ( dual1.primal * dual2.primal, dual1.partials * dual2.primal + dual2.partials * dual1.primal );
}


template < typename T >
Enable_if<dual_impl::can_become_dual_v<T>, NDual<T> > 
get_div_numerator( const NDual<T>& dual1, const NDual<T>& dual2 )
{
	return ( dual1.partials * dual2.primal - dual2.partials * dual1.primal );
}

template <typename T>
Enable_if<dual_impl::can_become_dual_v<T>, NDual<T> > 
operator/(const NDual<T>& dual1, const NDual<T>& dual2)
{
	return ( dual1.primal / dual2.primal, get_div_numerator( dual1,dual2 )/(dual2.primal * dual2.primal) );
}






template <typename T> inline constexpr        T  value   ( T x )               { return x; }
template <typename T> inline constexpr        T  value   ( const NDual<T>& x ) { return x.value(); }
template <typename T> inline constexpr Vector<T> partials( const NDual<T>& x ) { return x.partials; }










//need to define zero(Vector<T>) for use vvi








/*
#ifndef DUAL_H
#define DUAL_H

#include <iostream>
using namespace std;

template<typename Scalar>
class DualNumber
{
public:
	DualNumber(const Scalar& realPart, const Scalar& gradientPart = Scalar())
		:mReal(realPart), mDual(gradientPart)
	{
	}

	inline Scalar getReal() const { return mReal; }
	inline Scalar getDual() const { return mDual; }
	inline void print() { std::cout << mReal << "+" << mDual << "j" << std::endl; }
	//friend DualNumber<Scalar> operator+(const DualNumber<Scalar>& lhs, const DualNumber<Scalar>& rhs);
	friend DualNumber<Scalar> operator * (const DualNumber<Scalar>& lhs, const DualNumber<Scalar>& rhs);
	friend DualNumber<Scalar> operator / (const DualNumber<Scalar>& lhs, const DualNumber<Scalar>& rhs);
	friend DualNumber<Scalar> operator - (const DualNumber<Scalar>& lhs, const DualNumber<Scalar>& rhs);
private:
	Scalar mReal;
	Scalar mDual;
};


template<typename Scalar>
DualNumber<Scalar> makeDualNumber(const Scalar& realPart, const Scalar& dualPart = Scalar())
{
	return DualNumber<Scalar>(realPart, dualPart);
}

template<typename Scalar>
DualNumber<Scalar> operator+(const DualNumber<Scalar>& lhs, const DualNumber<Scalar>& rhs)
{
	Scalar r1 = lhs.getReal();
	Scalar d1 = lhs.getDual();
	Scalar r2 = rhs.getReal();
	Scalar d2 = rhs.getDual();
	return makeDualNumber(r1 + r2, d1 + d2);
}

template<typename Scalar>
DualNumber<Scalar> operator * (const DualNumber<Scalar>& lhs, const DualNumber<Scalar>& rhs)
{
	Scalar r1 = lhs.getReal();
	Scalar d1 = lhs.getDual();
	Scalar r2 = rhs.getReal();
	Scalar d2 = rhs.getDual();
	return makeDualNumber(r1 * r2, r1 * d2 + r2 * d1);
}

template<typename Scalar>
DualNumber<Scalar> operator - (const DualNumber<Scalar>& lhs, const DualNumber<Scalar>& rhs)
{
	Scalar r1 = lhs.getReal();
	Scalar d1 = lhs.getDual();
	Scalar r2 = rhs.getReal();
	Scalar d2 = rhs.getDual();
	return makeDualNumber(r1 * r2, r1 * d2 + r2 * d1);
}

template<typename Scalar>
DualNumber<Scalar> operator / (const DualNumber<Scalar>& lhs, const DualNumber<Scalar>& rhs)
{
	Scalar r1 = lhs.getReal();
	Scalar d1 = lhs.getDual();
	Scalar r2 = rhs.getReal();
	Scalar d2 = rhs.getDual();
	return makeDualNumber(r1 / r2, (d1 * r2 - r1 * d2) / (r2 * r2));
}

#endif

#include <iostream>
using namespace std;
template<typename Scalar>
Scalar testFunc(Scalar x)
{
	return (x * x) / (x + Scalar(1));
}
void test() {
	std::cout << testFunc<DualNumber<float>>(makeDualNumber(2.f, 1.f)).getDual() << std::endl;

}

*/

/*
 *


#include <iostream>
#include <functional>
#include "Dual.hpp"

using namespace std;


typedef float S;
typedef DualNumber<S> Dual;

template<typename F, typename... params>
Dual derivative(F f, params... vars) {
	auto val = f(vars...);
	return val;

}

//template<typename S>
//class Derivative {
//public:
//	Derivative(function<DualNumber<S>(DualNumber<S>, DualNumber<S>)> func) {
//		f = func;
//		f_defined = true;
//	}
// 
//	template<typename F>
//	Derivative(F&& func) {
//		f = func;
//		f_defined = true;
//	}
//
//	void checkF() {
//		cout << f_defined << endl;
//	}
//
//	template<typename... params>
//	DualNumber<S> at(params... vars) {
//		auto val = f(vars...);
//		return val;
//
//	}
//
//private:
//	bool f_defined = false;
//
//};


Dual testFunction(Dual x, Dual y) {
	return x * y;
}

Dual testFunction2(Dual x, Dual y, Dual z) {
	return x * y * z;
}

int main()
{
	
	
	// duals
	Dual d1 = makeDualNumber(3.f, 1.f) ;
	Dual d2 = makeDualNumber(1.f, 1.f);
	Dual d3 = makeDualNumber(1.f, 1.f);

	//Derivative<S> derivative = Derivative<S>(testFunction2<S>);
	//auto result = derivative.at(d1, d1, d3);
	//result.print();
	Dual result = derivative(testFunction2, d1, d2, d3);
	result.print();
}
*/


} //namespace linalg


namespace std
{

template <class T>
struct numeric_limits<linalg::NDual<T>> : numeric_limits<T>
{
	static constexpr bool is_specialized = true;
	static constexpr linalg::NDual<T> lowest()       { return numeric_limits<T>::lowest(); }
	static constexpr linalg::NDual<T> epsilon()      { return numeric_limits<T>::epsilon(); }
	static constexpr linalg::NDual<T> round_error()  { return numeric_limits<T>::round_error(); }
	static constexpr linalg::NDual<T> infinity()     { return numeric_limits<T>::infinity(); }
	static constexpr linalg::NDual<T> quiet_NaN()    { return numeric_limits<T>::quiet_NaN(); }
	static constexpr linalg::NDual<T> signaling_NaN(){ return numeric_limits<T>::signaling_NaN(); }
	static constexpr linalg::NDual<T> denorm_min()   { return numeric_limits<T>::denorm_min(); }
};



}//namespace std;

#endif
