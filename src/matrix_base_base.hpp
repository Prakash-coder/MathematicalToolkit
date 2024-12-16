#ifndef BASE_BASE_HPP
#define BASE_BASE_HPP

#include <cassert>
#include <iostream>

#include "matrix_slice.hpp"
#include "utility.hpp"
#include "matrix_impl.hpp"

namespace linalg
{

template <typename T, size_t N>
class TensorBase
{
public:
	typedef T value_type;
	static constexpr size_t order = N;
	
	TensorBase()                               {};  //default
	TensorBase( TensorBase&& )                 {}; //move
	TensorBase& operator=( TensorBase&& )      {};  //move
	TensorBase( const TensorBase& )            {}; //copy
	TensorBase& operator=( const TensorBase& ) {}; //copy
	~TensorBase() {}; //virtual destructor

	template <typename... Exts>
	explicit TensorBase( Exts... exts) 
		:descriptor{exts...} 
	{}

	explicit TensorBase( const TensorSlice<N>& slice ) 
		:descriptor{slice}
	{}

	static constexpr size_t n_dims() { return order; } //vvi available at compile time as after ::operator

	size_t extents( size_t n ) const { assert(n<order); return descriptor.extents[n]; }


	const TensorSlice<N>& Descriptor() const { return descriptor; }

	virtual size_t   size()       = 0;
	virtual       T* data()       = 0;
	virtual const T* data() const = 0;

	size_t n_rows() const { return descriptor.extents[0]; }
	size_t n_cols() const { return descriptor.extents[1]; }

	template <typename... Args>
	Enable_if< RequestingElement<Args...>(), T& > operator()( Args... args );

	template <typename... Args>
	Enable_if< RequestingElement<Args...>(), const T& > operator()( Args... args ) const;

protected:
	TensorSlice<N> descriptor;
};

template < typename T, size_t N >
template <typename... Args>
Enable_if< RequestingElement<Args...>(), T& > TensorBase<T,N>::operator()( Args... args )
{ 
	assert( check_bounds( this->descriptor, args... ) );
	return *( data() + 	this->descriptor( args... ));
}

template < typename T, size_t N>
template <typename... Args>
Enable_if< RequestingElement<Args...>(), const T& > TensorBase<T,N>::operator()( Args... args ) const
{
	assert( check_bounds( this->descriptor, args... ) );
	return *( data() + this->descriptor( args... ) );
}


template <typename T, size_t N>
std::ostream& operator<<( std::ostream& os, const TensorBase<T,N>& tbase )
{
	os << '{';
	for ( auto i = 0; i != tbase.n_rows(); ++i )
	{
		os << tbase[i];
		if ( i + 1 != tbase.n_rows() )
		{
			os << ',';
		}
	}
	return os << '}';
}



} //namespace linalg;
#endif
