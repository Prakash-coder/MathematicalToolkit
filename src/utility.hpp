#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <type_traits>
#include <functional>
#include <numeric>
#include <chrono>

namespace linalg
{

template <size_t N>
struct TensorSlice;

template <typename T, size_t N>
struct Tensor_init; 

template <typename T, size_t N>
class TensorBase;
//forward declaration for no compiler/linker errors
template <typename T, size_t N>
class Tensor;

template <typename T, size_t N>
class TensorRef;


template < typename T >
class NDual;


namespace dual_impl
{
/*---------------------------------x---------------------------------------------*/
//refactored this from dualad.hpp to make use of metafunctions so it looks more oop like
//generalization
template<typename T> struct can_become_dual    : std::false_type{ };
//specializations
template<> struct can_become_dual<double>      : std::true_type { };
template<> struct can_become_dual<float>       : std::true_type { };
template<> struct can_become_dual<long double> : std::true_type { };
template<> struct can_become_dual<int>         : std::true_type { };
template<> struct can_become_dual<unsigned>    : std::true_type { };
template<> struct can_become_dual<long long>   : std::true_type { };
template<> struct can_become_dual<short>       : std::true_type { };
template<> struct can_become_dual<size_t>      : std::true_type { };

template < typename T > struct is_dual            :std::false_type { };
template < typename T > struct is_dual< NDual<T> >:std::true_type  { };

//type alias
template <typename T> constexpr bool can_become_dual_v = can_become_dual<T>::value;









//to be called as can_become_dual_v<T> to return a bool.
/*---------------------------------x---------------------------------------------*/
}
//aliases for type_traits 
/*---------------------------------x---------------------------------------------*/

template < bool B, typename T = void >
using Enable_if = typename std::enable_if<B,T>::type;


template <typename T>
using Value_type = 	typename T::value_type;

template <typename... PARAMS>
using common_type_t = typename std::common_type_t<PARAMS...>::type;




template < typename X, typename Y > constexpr bool is_same_v = std::is_same<X,Y>::value;

template <typename T, typename F>
constexpr bool Same()
{
	return std::is_same<T,F>::value;
}

constexpr bool All() { return true; }

template < typename... Args >
constexpr bool All( bool b, Args... args ) 
{
	return b && All( args... );
}

template <typename... Args>
constexpr bool Some() { return false; }

template <typename... Args>
constexpr bool Some( bool b, Args... args )
{
	return b || Some(args...);
}

template <typename T>
void ignore( T&& ) {}

struct sub_fail {};

template <typename T>
struct sub_pass : std::true_type {};

template <>
struct sub_pass<sub_fail> : std::false_type {};

template <typename M>
struct get_Tensor_type
{
	template <typename T, size_t N, typename = Enable_if<(N >= 1)> >
	static bool check(const Tensor<T,N>& m);

	static sub_fail check(...);

	using  type = decltype(check(std::declval<M>()));
};

template <typename T>
struct has_Tensor_type :sub_pass< typename get_Tensor_type<T>::type>
{};

template <typename M>
constexpr bool Has_Tensor_Type()
{
	return has_Tensor_type<M>::value;
}

template <typename M>
using Tensor_type_result = typename get_Tensor_type<M>::type;

template <typename M>
constexpr bool Tensor_type()
{
	return Has_Tensor_Type<M>();
}

template <typename C>
using Value_Type = typename C::value_type;

template <typename T1, typename T2>
constexpr bool Convertible()
{
	return std::is_convertible<T1,T2>::value;
}

/*---------------------------------x---------------------------------------------*/



/*
 * The following are a set of functors for the various assign operations for 
 * our tensors/matrices.
*/


//The following arithmetic operations are assignment operations
template <typename T> struct assign_to_val{ constexpr void operator()( T& lhs, const T& rhs ){ lhs  = rhs; } }; 
template <typename T> struct add_op_assign{ constexpr void operator()( T& lhs, const T& rhs ){ lhs += rhs; } };
template <typename T> struct sub_op_assign{ constexpr void operator()( T& lhs, const T& rhs ){ lhs -= rhs; } };
template <typename T> struct mul_op_assign{ constexpr void operator()( T& lhs, const T& rhs ){ lhs *= rhs; } };
template <typename T> struct div_op_assign{ constexpr void operator()( T& lhs, const T& rhs ){ lhs /= rhs; } };
template <typename T> struct mod_op_assign{ constexpr void operator()( T& lhs, const T& rhs ){ lhs %= rhs; } };
template <typename T> struct negate_all   { constexpr void operator()( T& lhs){ lhs = -lhs; } };


template <typename T>
inline
Tensor<T,2> Transpose( const Tensor<T,1>& x )
{
	Tensor<T,2> res( 1, x.n_rows() );
	std::copy( x.begin(), x.end(), res.begin() );
	return res;
}

template <typename T>
inline
Tensor<T,2> Transpose( const TensorRef<T,1>& x )
{
	Tensor<T,2> res( 1, x.n_rows() );
	std::copy( x.begin(), x.end(), res.begin() );
	return res;
}

template < typename T > 
inline 
Tensor<T,2> Transpose( const Tensor<T,2>& x )
{
	Tensor<T,2> res( x.n_cols(), x.n_rows() );
	for ( auto i = 0; i < x.n_rows(); ++i )
	{
		res.col(i) = x.row(i);
	}
	return res;
}

template <typename T>
inline 
Tensor<T, 2> Transpose(const TensorRef<T, 2> &a) 
{
	Tensor<T, 2> res(a.n_cols(), a.n_rows());
	for (std::size_t i = 0; i < a.n_rows(); ++i) 
	{
		res.col(i) = a.row(i);
    }

    return res;
}

template <typename T, size_t N, typename... Args>
inline 
auto reshape ( const Tensor<T,N>& x, Args... args )
	-> decltype( Tensor<T, sizeof...(args)>())
{
	Tensor<T, sizeof...(args)> res(args...);
	std::copy( x.begin(), x.end(), res.begin() );
	return res;
}

template <typename T>
inline
T prod( const Tensor<T,1>& x )
{
	return std::accumulate( x.begin(), x.end(), T{1}, std::multiplies<T>() );
}

template <typename T>
inline
T prod( const TensorRef<T,1>& x )
{
	return std::accumulate( x.begin(), x.end(), T{1}, std::multiplies<T>() );
}

template < typename T, size_t N >
T as_scalar( const Tensor<T,N>& x )
{
	assert(x.size() == 1);
	return *(x.data());
}


template <typename M, typename... Args>
Enable_if< Tensor_type<M>(), M > zeros( Args... args )
{
	assert(M::order() == sizeof...(args));
	M res(args...);
	res = 0;
	return res;
}

template <typename T>
inline 
Tensor<T,2> diagonal( const Tensor<T,1>& x )
{
	Tensor<T,2> temp( x.size(), x.size() );
	temp.diag() = x;
	return temp;
}

template <typename M>
Enable_if< Tensor_type<M>(), M > eye( size_t i, size_t j)
{
	assert(M::n_dims() == 2);
	M res(i,j);
	res.diag() = 1;
	return res;
}

template <typename T>
inline T dot(const TensorBase<T, 1> &a, const TensorBase<T, 1> &b) 
{
	assert( a.size() == b.size() );	
	T res = T{0};
	for ( size_t n = 0; n != a.size(); ++n) 
	{
		res += a(n) * b(n);
	}
	return res;
}








}//namespace linalg





#endif
