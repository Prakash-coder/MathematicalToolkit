#ifndef MATRIX_INITIALIZER_H
#define MATRIX_INITIALIZER_H

#include <initializer_list>
#include <iostream>
namespace linalg
{


template <typename T, size_t N>
struct Tensor_init
{
	using type = std::initializer_list< typename Tensor_init<T,N-1>::type>;
};


template <typename T>
struct Tensor_init<T,1>
{
	using type = std::initializer_list<T>;
};

template < typename T >
struct Tensor_init<T,0>;

template <typename T, size_t N>
using TensorInitializer = typename Tensor_init<T,N>::type;




} //namespace linalg


#endif 
