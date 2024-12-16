#ifndef MATRIX_IMPL_H
#define MATRIX_IMPL_H


#include <initializer_list>

//#include "matrix_ref.hpp"
#include "utility.hpp"


namespace linalg
{


struct slice
{
	slice() 
		:start(-1)
		,length(-1)
		,stride(1)
	{}

	explicit slice( size_t s ) 
		:start(s)
		,length(-1)
		,stride(1)
	{}
	
	slice ( size_t s, size_t l, size_t n =1 ) 
		:start(s)
		,length(l)
		,stride(n)
	{}

	size_t operator()( size_t i ) const { return start + i * stride; }

	static slice all;



	size_t start;
	size_t length;
	size_t stride;
};



template <size_t N>
struct TensorSlice;

//some forward declarations cause why not



template < typename... Args >
bool RequestingElement( Args... args );

template < typename... Args >
bool RequestingSlice( Args... args );



template <typename List>
bool check_not_non_rectangular( const List& list )
{
	auto i = list.begin();	
	for ( auto j = i+1; j!=list.end(); ++j )
	{
		if ( i->size()!=j->size() )
		{
			return false;
		}
	}
	return true;
}

template <size_t N, typename T, typename List>
Enable_if< (N>1), void >  add_extents( T& first, const List& list )
{
	assert( check_not_non_rectangular(list) );
	*first = list.size();
	add_extents< N-1 >(++first, *list.begin() );
}

template <size_t N, typename T, typename List>
Enable_if< (N==1), void > add_extents( T& first, const List& list )
{
	*first++ = list.size();
}



template < size_t N, typename List >
std::array<size_t, N> derive_extents( const List& list )
{
	std::array<size_t, N> a;
	auto f = a.begin();
	add_extents<N>(f,list);
	return a;
}

template <size_t N>
size_t compute_strides( const std::array<size_t, N >& extents, std::array<size_t, N>& strs )
{
	size_t st = 1;
	for ( auto i = N-1; i >= 0; --i )
	{
		strs[i] = st;
		st *= extents[i];
	}
	return st;
}

template <size_t N>
size_t compute_size( const std::array<size_t, N> &exts )
{
  return std::accumulate( exts.begin(), exts.end(), 1, std::multiplies<size_t>{} );
}



template <typename T, typename Vec>
void add_list( const T* first, const T* last, Vec& vec)
{
	vec.insert( vec.end(), first, last );
}


template<typename T, typename Vec> // nested initializer_lists
void add_list(const std::initializer_list<T>* first, const std::initializer_list<T>* last, Vec& vec)
{
	for ( ; first!=last; ++first )
	{
		add_list( first->begin(), first->end(), vec );
	}
}

template <typename T, typename Vec>
void insert_flat( std::initializer_list<T> list, Vec& vec )
{
	add_list(list.begin(), list.end(), vec);
}

template <typename T, typename Iter>
void copy_list( const T *first, const T *last, Iter &iter )
{
  iter = std::copy( first, last, iter );
}

template <typename T, typename Iter>
void copy_list(const std::initializer_list<T> *first, const std::initializer_list<T> *last, Iter &it) 
{
	for (; first != last; ++first) 
	{
		copy_list( first->begin(), first->end(), it );
	}
}

template <typename T, typename Iter>
void copy_flat( std::initializer_list<T> list, Iter &iter ) 
{
	copy_list( list.begin(), list.end(), iter );
}

template <std::size_t I, std::size_t N>
void slice_dim( size_t Offset, const TensorSlice<N> &desc, TensorSlice<N - 1> &row )
{
	row.offset = desc.offset;
	int j = (int)N - 2;
	for (int i = N - 1; i >= 0; --i) 
	{
		if (i == I)
		{
			row.offset += desc.strides[i] * Offset;
		}
        else 
        { 
			row.extents[j] = desc.extents[i];
			row.strides[j] = desc.strides[i];
			--j;
        }
    }
	row.size = compute_size(row.extents);
}

template <typename... Args>
constexpr bool Requesting_element() {
  return All(Convertible<Args, size_t>()...);
}

template <typename... Args>
constexpr bool Requesting_slice() {
  return All((Convertible<Args, size_t>() || Same<Args, slice>())...) &&
         Some(Same<Args, slice>()...);
}

template <size_t N, typename... Dims >
bool check_bounds( const TensorSlice<N> &slice, Dims... dims)
{
	size_t indices[N]{ size_t(dims)... };
	return std::equal( indices, indices + N, slice.extents.begin(), std::less<size_t>{});
}





template <std::size_t M, std::size_t N>
std::size_t do_slice_dim( const TensorSlice<N> &slice1, TensorSlice<N> &slice2, std::size_t s) 
{
    std::size_t i = N - M;
    slice2.strides[i] = slice1.strides[i];
    slice2.extents[i] = 1;
    return s * slice2.strides[i];
}

template <std::size_t NRest, std::size_t N>
std::size_t do_slice_dim( const TensorSlice<N> &os, TensorSlice<N> &ns, slice s )
{
    std::size_t i = N - NRest;
    ns.strides[i] = s.stride * os.strides[i];
    ns.extents[i] = (s.length == size_t(-1))
                               ? (os.extents[i] - s.start + s.stride - 1) / s.stride
                               : s.length;
    return s.start * os.strides[i];
}

template <std::size_t N>
std::size_t do_slice_dim2(const TensorSlice<N> &os, TensorSlice<N> &ns, slice s, std::size_t M ) 
{
    std::size_t i = N - M;
    ns.strides[i] = s.stride * os.strides[i];
    ns.extents[i] = (s.length == size_t(-1))
                               ? (os.extents[i] - s.start + s.stride - 1) / s.stride
                               : s.length;
    return s.start * os.strides[i];
}

template <std::size_t N>
std::size_t do_slice(const TensorSlice<N> &os, TensorSlice<N> &ns) 
{
	ignore(os);
	ignore(ns);
    return 0;
}

template <std::size_t N, typename T, typename... Args>
std::size_t do_slice(const TensorSlice<N> &os, TensorSlice<N> &ns, const T &s, const Args &... args) 
{
    std::size_t m = do_slice_dim<sizeof...(Args) + 1>(os, ns, s);
    std::size_t n = do_slice(os, ns, args...);
    return m + n;
}

//template < typename T, size_t N>
//TensorRef<T,N> row(size_t n );


//ToDo
//check_bounds
//derive_extents()
//insert_flat
//Tensor_type<>

//compute_strides computes strides by taking two std::array as inputs
//compute_size
//slice_dim
//all of these need to be in this file 







} //namespace linalg
#endif
