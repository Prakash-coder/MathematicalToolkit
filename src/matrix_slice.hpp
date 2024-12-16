#ifndef MATRIX_SLICE_H
#define MATRIX_SLICE_H

#include <iostream>
#include <array>
//#include <algorithm>
#include <initializer_list>
#include <cassert>
#include <numeric> //for std::inner_product()
#include "matrix_impl.hpp"

//#include "matrix_impl.hpp"
//#include "matrix_impl.hpp"
namespace linalg

{

/* 
 * TensorSlice contains the details about the starting offset, the 
 * side of the sub-array, the extents of each dimension and the strides::offset between
 * elements in each dimension
 */




template < size_t N >
struct TensorSlice
{
	TensorSlice();	 //constructor

	TensorSlice( size_t s, std::initializer_list< size_t > exts ); //extents

	TensorSlice( size_t s, 
				 std::initializer_list< size_t > exts, 
				 std::initializer_list< size_t > strds 
				 ); //extents and strides

	TensorSlice( const std::array< size_t , N >& exts);

	template < typename... Dims >
	TensorSlice( Dims... dims ); //this construtor takes in the N extents of the N-dimensional matrix

	template< typename... Dims >
	size_t operator()(Dims... dims ) const;

	void clear();

	size_t Offset( const std::array<size_t, N>& pos ) const;

	
	size_t size; 					  //total number of elements
	size_t offset;						//starting offset
	std::array< size_t, N > extents;  //the vector containing the extents on each dimension, must have the same number of elements as the dimensionality
	std::array< size_t, N > strides; //offset between elements in each dimension

};

template < size_t N >
TensorSlice< N >::TensorSlice() 
	:size{1},
	 offset{0}
{
	std::fill( extents.begin(), extents.end(), 0 );
	std::fill( strides.begin(), strides.end(), 1 );
}

template < size_t N >
TensorSlice< N >::TensorSlice( size_t s, 
							   std::initializer_list< size_t > exts )
	:offset{s}
{
	assert( exts.size() == N );
	std::copy( exts.begin(), exts.end(), extents.begin() );
	size = compute_strides( extents, strides );
}

template < size_t N >
TensorSlice< N >::TensorSlice( size_t s, 
							   std::initializer_list< size_t > exts, 
							   std::initializer_list< size_t > strds )
	:offset{s}
{
	assert( exts.size() == N );
	std::copy( exts.begin(), exts.end(), extents.begin() );
	std::copy( strds.begin(), strds.end(), strides.begin() );
	size = compute_size( extents );
}

template < size_t N >
TensorSlice< N >::TensorSlice( const std::array< size_t , N >& exts)
	:offset{0},
	extents{exts} 
{
	assert( exts.size() == N );
	size = compute_strides( extents, strides );
}

template <size_t N>
size_t TensorSlice<N>::Offset( const std::array< size_t, N>& pos) const 
{
	assert( pos.size() == N );
	return offset + std::inner_product( pos.begin(), pos.end(), strides.begin(), size_t{0});
}

/*vvi calculates index in the flat from the set of subscripts
|
V
*/

template < size_t N >
template < typename... Dims >
TensorSlice< N >::TensorSlice( Dims... dims )
	:offset{0}
{
	static_assert( sizeof...(Dims) == N, "DimensionMismatchError:"
				   "TensorSlice( Dims... ), sizeof...(Dims) != N"
				 );
	std::array< size_t, N > args = { size_t( dims )... };
	std::copy( args.begin(), args.end(), extents.begin() );
	size = compute_strides( extents, strides );
}

template < size_t N >
template < typename... Dims >
size_t TensorSlice< N >::operator()( Dims... dims ) const
{
	static_assert( sizeof...(Dims) == N, "DimensionMismatchError:"
				   "TensorSlice( Dims... ), sizeof...(Dims) != N"
				 );
	std::array< size_t, N >	args = { size_t( dims )... };
	return offset + std::inner_product( args.begin(), args.end(), strides.begin(), size_t{0} );
//std::inner_product( inputiteratorfirst begin, inputiteratorfirst end, inputiteratorlast begin, T init );
}

template < size_t N >
inline bool equal_extent( const TensorSlice<N>& a, const TensorSlice<N>& b )
{
	return a.extents == b.extents;
}

template < size_t N >
void TensorSlice<N>::clear() 
{
	size = 0;
	offset = 0;
	extents.fill(0);
	strides.fill(0);
}

template<size_t N>
std::ostream& operator<<( std::ostream& os, const std::array<size_t, N>& array )
{
	for ( auto x : array )
	{
		os << x << ' ';
	}
	return os;
}


template <size_t N >
std::ostream& operator<<(std::ostream& os, const TensorSlice<N>& tensor)
{
	os << "size: " << tensor.size << ", start: " << tensor.offset <<
		", extents: " << tensor.extents << ", strides: " << tensor.strides;
	return os;
}

template <size_t N>
inline bool operator==(const TensorSlice<N>& a, const TensorSlice<N>& b)
{
	return a.offset == b.offset 
		   && std::equal( a.extents.cbegin(), a.extents.cend(), b.extents.cbegin() )
		   && std::equal( a.strides.cbegin(), a.strides.cend(), b.strides.cbegin() );
}

template <size_t N>
inline bool operator!=(const TensorSlice<N>& a, const TensorSlice<N>& b)
{
	return !(a==b);
}


} //namespace linalg





#endif
