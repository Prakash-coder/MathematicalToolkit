#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <cassert>


#include "linalg.hpp"
#include "exceptions.hpp"
#include "matrix_base_base.hpp"
#include "matrix_slice.hpp"
#include "matrix_ref.hpp"
#include "matrix_initializer.hpp"
#include "matrix_impl.hpp"
#include "utility.hpp"

//#include <stdexcept>
namespace linalg
{

/*
 * The implementation of the matrix and everything related to it is heavily inspired from 
Bjarne Stroustroup's book The C++ Programming Language Chapter 29: A Matrix Design
*/



/* 
 * Tensor<T,N> is an N-dimensional tensor containing elements of type T
 * stored in the flat, linear memory of std::vector<T>.
 */

template < typename T, size_t N >
class Tensor : public TensorBase<T,N> 
{

public:

	static constexpr size_t dimensionality = N;
	using value_type                       = T;
	using iterator                         = typename std::vector<T>::iterator;
	using const_iterator                   = typename std::vector<T>::const_iterator;
	


	//constructors
	
	Tensor()                           = default; //default constructor;
	Tensor( Tensor&& )                 = default; //move constructor;
	Tensor& operator=( Tensor&& )      = default;   //move assignment operator;
	Tensor( const Tensor& )            = default; //copy constructor;
	Tensor& operator=( const Tensor& ) = default; //copy assignment operator;
	~Tensor()                          = default; //destructor



	template < typename M, typename =Enable_if<Tensor_type<M>()>>	
	Tensor( const M& x );
	
	template < typename M, typename =Enable_if<Tensor_type<M>()>>	
	Tensor& operator=( const M& x );

//TensorRef is a reference to a submatrix
	template < typename U >
	Tensor( const TensorRef< U , N >& );

	template < typename U >
	Tensor& operator=( const TensorRef< U, N >& x );

	template < typename... Exts >
	explicit Tensor( Exts... exts ); //extents

	// to initialize or assign our tensor from an initializer list
	Tensor( TensorInitializer< T, N > init );

	Tensor& operator=( TensorInitializer< T, N > );

	template < typename U, size_t NN = N,
			   typename = Enable_if<Convertible<U,size_t>()>, 
			   typename = Enable_if <(NN>1)> >
	Tensor( std::initializer_list< U > )            = delete;

	template < typename U, size_t NN = N,
			   typename = Enable_if<Convertible<U,size_t>()>, 
			   typename = Enable_if <(NN>1)> >
	Tensor& operator=( std::initializer_list< U > ) = delete;
	// We want to specify the extents in each dimension of our matrix through this Tensor  

	static constexpr size_t order() { return dimensionality; } //available to others as Matrix<T,N>::order() hence static


	size_t size() { return elements.size(); }


	//virtual size_t size() const = 0; //pure virtual function



	//flat access of elements
	T* data() { return elements.data(); }
	const T* data() const { return elements.data(); }


public :
	using TensorBase<T,N>::operator();


//tensor1(i,j,k) subscripting with integers.

	template < typename... Args >
	Enable_if< RequestingSlice< Args... >(), TensorRef< T, N > > 
	operator()( const Args&... args );

	template < typename... Args >
	Enable_if< RequestingSlice< Args... >(), TensorRef< const T, N > > 
	operator()( const Args&... args ) const;
	

	//needs some more thinking

	TensorRef< T, N-1 >       operator[]( size_t n ) { return row(n); }
	TensorRef< const T, N-1 > operator[]( size_t n ) const { return row(n); }
	
	TensorRef< T, N-1 >       row( size_t n );
	TensorRef< const T, N-1 > row( size_t n ) const; 

	TensorRef< T, N-1 >       col( size_t n );
	TensorRef< const T, N-1 > col( size_t n ) const; 
	
	TensorRef<T,N>            rows( size_t i, size_t j );
	TensorRef<const T,N>      rows( size_t i, size_t j ) const;

	TensorRef<T,N>            cols( size_t i, size_t j );
	TensorRef<const T,N>      cols( size_t i, size_t j ) const;

	// for iterators:
	iterator begin() { return elements.begin(); }
	const_iterator begin() const { return elements.cbegin(); }
	iterator end() { return elements.end(); }
	const_iterator end() const { return elements.cend(); }




	//apply  functions for later use
	template < typename F >
	constexpr Tensor<T,N>& apply_to_each_element( F f );
	
	template < typename F >
	constexpr Tensor< T, N >& apply_to_each_element( T& c, F f );

	template < typename F, typename M >
	constexpr Tensor<T,N>& apply_to_each_element( M& m, F f );



	// assignment operations on tensors-scalar
	Tensor& operator=( const T& scalar );
	Tensor& operator+=( const T& scalar );
	Tensor& operator-=( const T& scalar );
	Tensor& operator*=( const T& scalar );
	Tensor& operator/=( const T& scalar );
	Tensor& operator%=( const T& scalar );
	
	//assignment operations on tensors-tensors
	Tensor<T,N>& operator+=( const Tensor<T,N>& tensor2 );
	Tensor<T,N>& operator-=( const Tensor<T,N>& tensor2 );
	Tensor<T,N>& operator*=( const Tensor<T,N>& tensor2 );
	Tensor<T,N>& operator/=( const Tensor<T,N>& tensor2 );
	Tensor<T,N>& operator%=( const Tensor<T,N>& tensor2 );

	template <typename M>
	Enable_if<Tensor_type<M>(), Tensor& > 	operator+=( const M& x );

	template <typename M>
	Enable_if<Tensor_type<M>(), Tensor& > 	operator-=( const M& x );

	template <typename M>
	Enable_if<Tensor_type<M>(), Tensor& > 	operator*=( const M& x );


	template <typename M>
	Enable_if<Tensor_type<M>(), Tensor& > 	operator/=( const M& x );


	template <typename M>
	Enable_if<Tensor_type<M>(), Tensor& > 	operator%=( const M& x );


	template <typename U = typename std::remove_const<T>::type > //unary - operator
	Tensor<U,N> operator-() const;




	//binary operators like +, - etc are non-members
//vector from matrix
	template <typename U, std::size_t NN = N, typename = Enable_if<(NN == 1)>>
	Tensor(const Tensor<U, 2> &x);

	template <typename U, std::size_t NN = N, typename = Enable_if<(NN == 1)>>
	Tensor(const TensorRef<U, 2> &x);

	template <typename U, std::size_t NN = N, typename = Enable_if<(NN == 1)>>
	Tensor &operator=(const Tensor<U, 2> &x);

	template <typename U, std::size_t NN = N, typename = Enable_if<(NN == 1)>>
	Tensor &operator=(const TensorRef<U, 2> &x);

//matrix from vector
 	template <typename U, std::size_t NN = N, typename = Enable_if<(NN == 2)>>
	Tensor(const Tensor<U, 1> &x);

	template <typename U, std::size_t NN = N, typename = Enable_if<(NN == 2)>>
	Tensor(const TensorRef<U, 1> &x);

	template <typename U, std::size_t NN = N, typename = Enable_if<(NN == 2)>>
	Tensor &operator=(const Tensor<U, 1> &x);

	template <typename U, std::size_t NN = N, typename = Enable_if<(NN == 2)>>
	Tensor &operator=(const TensorRef<U, 1> &x);



	//return std::vector from a Tensor
	template < size_t NN = N, typename = Enable_if<( NN==1 )> >
	std::vector<T> GetTostdVec() const { return std::vector<T>( begin(), end() ); }

	//vector from matrix

	
	template < size_t NN = N, typename = Enable_if<(NN==1) || (NN==2)> >
	Tensor<T,2> transpose() const { return Transpose( *this ); }

//diagonal elements
	template < size_t NN = N, typename = Enable_if<(NN==2)>>
	TensorRef<T,1> diag()
	{
		assert(this->n_rows() == this->n_cols());
		TensorSlice<1> d;
		d.offset = this->descriptor.offset;
		d.extents[0] = this->n_rows();
		d.strides[0] = this->n_rows() + 1;
		return ( d, data() );
	}
	
	template <std::size_t NN = N, typename = Enable_if<(NN == 2)>>
	TensorRef<const T, 1> diag() const 
	{
		assert(this->n_rows() == this->n_cols());
		TensorSlice<1> d;
		d.offset     = this->descriptor.offset;
		d.extents[0] = this->n_rows();
		d.strides[0] = this->n_rows() + 1;
		return (d, data());
  }

	template <typename F>
	Tensor& apply(F f);

	template <typename M, typename F>
	Enable_if<Tensor_type<M>(), Tensor& > apply( const M& m, F f);

	bool is_empty() const { return begin() == end(); }
	void clear();

protected:

	std::vector< T > elements;			//the elements to be stored in the tensor are stored flat in the vector elements.
};

template < typename T, size_t N >
template <typename M, typename X>
Tensor<T,N>::Tensor( const M& x )
	:TensorBase<T,N>( x.Descriptor(), elements(x.begin(), x.end() ))
{
	static_assert( Convertible<typename M::value_type, T>(),"");
}

template < typename T, size_t N >
template <typename M, typename X>
Tensor<T,N>& Tensor<T,N>::operator=( const M& x )
{
	static_assert( Convertible<typename M::value_type, T>(),"");
	this->descriptor = x.Descriptor();
	elements.assign( x.begin(), x.end() );
	return *this;
}

/*
 *Tensor<T,N>::Tensor( const TensorRef<U,N>& x ) is a constructor to take a TensorRef
 *object which is basically a reference to Tensor and can be seen as either the 
 * row or the column of a Tensor.
 */

template < typename T, size_t N >
template <typename U>
Tensor<T,N>::Tensor( const TensorRef<U,N>& x )
	:TensorBase<T,N>{ x.Descriptor().extents }
	,elements{ x.begin(), x.end() }
{
	static_assert(Convertible<U,T>(), "T,N type error.");
}

template < typename T, size_t N >
template < typename U >
inline 
Tensor< T, N >& Tensor< T, N >::operator=( const TensorRef< U, N >& x )
{
	static_assert( Convertible<U,T>(), "Incompatible Element Types" );
	this->descriptor = x.Descriptor();
	this->descriptor.offset = 0;
	elements.assign( x.begin(), x.end());
	return *this;
}
/*
 * Tensor<T,N>::Tensor( Exts... exts ) is a constructor of the tensor class that takes 
 * the 'extents' in each of the N axes/dimensions of the Tensor<T,N>.
 */
template < typename T, size_t N >
template < typename... Exts >
inline 
Tensor< T, N >::Tensor( Exts... exts ) 
	:TensorBase<T,N>{ exts... }
	,elements( this->descriptor.size )
{}


/*
 * Tensor<T,N>::Tensor( TensorInitializer<T,N> init ) is a Tensor constructor that 
 * takes the actual inputs to the Tensor of type T through a suitably nested 
 * intializer list wrapper TensorInitializer<T,N>.
 */
template < typename T, size_t N >
inline 
Tensor< T, N >::Tensor( TensorInitializer< T, N > init ) 
{
	this->descriptor.offset = 0;
	this->descriptor.extents = derive_extents<N>( init );
	this->descriptor.size = compute_strides( this->descriptor.extents, this-> descriptor.strides );
	elements.reserve( this->descriptor.size );
	insert_flat( init,elements );
	assert( elements.size() == this->descriptor.size );
}

template < typename T, size_t N >
inline 
Tensor< T, N >& Tensor< T, N >::operator=( TensorInitializer< T, N > init )
{
	elements.clear();
	this->descriptor.offset = 0;
	this->descriptor.extents = derive_extents<N>( init );
	this->descriptor.size = compute_strides( this->descriptor.extents, this->descriptor.strides );
	elements.reserve( this->descriptor.size );
	insert_flat( init, elements );
	assert( elements.size() == this-> descriptor.size );
	return *this;
}






//C-style subscripting through tensor1[i] giving the i-th row
template <typename T, size_t N>
template < typename... Args >
Enable_if< RequestingSlice< Args... >(), TensorRef< T, N > > 
Tensor<T,N>::operator()( const Args&... args )
{
	TensorSlice<N> d;
	d.offset       = do_slice( this->descriptor, d, args... );
	d.size         = compute_size(d.extents);
	return ( d, data() );
}

template <typename T, size_t N>
template < typename... Args >
Enable_if< RequestingSlice< Args... >(), TensorRef< const T, N > > 
Tensor<T,N>::operator()( const Args&... args ) const
{
	TensorSlice<N> d;
	d.offset = do_slice( this->descriptor, d, args...);
	d.size   = compute_size( d.extents );
	return { d, data() };
}



template < typename T, size_t N >
inline 
TensorRef< T, N-1 > Tensor< T, N >::row( size_t n )
{
	assert( n < this->n_rows() );
	TensorSlice< N-1 > row;
	slice_dim<0>( n, this->descriptor, row );
	return { row, data() };
}

template <typename T, size_t N>
TensorRef< const T, N-1 > Tensor<T,N>::row( size_t n ) const
{
	assert( n < this->n_rows() );
	TensorSlice<N-1> row;
	slice_dim<0>( n, this->descriptor, row );
	return { row, data() };
}


template <typename T, size_t N>
TensorRef<T,N-1> Tensor<T,N>::col( size_t n )
{
	assert( n < this->n_cols() );
	TensorSlice<N-1> column;
	slice_dim<1>( n, this->descriptor, column );
	return { column, data() };
}

template <typename T, size_t N>
TensorRef< const T,N-1> Tensor<T,N>::col( size_t n ) const
{
	assert( n < this->n_cols() );
	TensorSlice<N-1> column;
	slice_dim<1>( n, this->descriptor, column );
	return { column, data() };
}

template <typename T, size_t N>
TensorRef<T,N> Tensor<T,N>::rows( size_t i, size_t j)
{
	assert( i <= j && j < this->n_rows() );
	TensorSlice<N> d;
	d.offset = this->descriptor.offset;
	d.offset += do_slice_dim<N>( this->descriptor, d, slice{ i, j - i + 1 });
	size_t NRest = N - 1;
	while ( NRest >= 1 )
	{
		d.offset += do_slice_dim2( this-> descriptor, d, slice{0}, NRest );
		--NRest;
	}
	return { d, data() };
} 


template <typename T, size_t N>
TensorRef< const T,N > Tensor<T,N>::rows( size_t i, size_t j) const
{
	assert( i <= j && j < this->n_rows() );
	TensorSlice<N> d;
	d.offset = this->descriptor.offset;
	d.offset += do_slice_dim<N>( this->descriptor, d, slice{ i, j - i + 1 });
	size_t NRest = N - 1;
	while ( NRest >= 1 )
	{
		d.offset += do_slice_dim2( this-> descriptor, d, slice{0}, NRest );
		--NRest;
	}
	return { d, data() };
} 

template <typename T, std::size_t N>
TensorRef<T, N> Tensor<T, N>::cols( size_t i, size_t j ) 
{
	assert(N >= 2 && i <= j && j < this->n_cols() );

	TensorSlice<N> d;
	d.offset = this->desc_.start;
	d.offset += do_slice_dim<N>(this->desc_, d, slice{0});
	d.offset += do_slice_dim<N - 1>(this->desc_, d, slice{i, j - i + 1});
	
	std::size_t NRest = N - 2;
	while (NRest >= 1) {
	  d.start += do_slice_dim2(this->desc_, d, slice{0}, NRest);
	  --NRest;
	}
	return {d, data()};
}


template <typename T, std::size_t N>
TensorRef<const T, N> Tensor<T, N>::cols( size_t i, size_t j ) const
{
	assert(N >= 2 && i <= j && j < this->n_cols() );

	TensorSlice<N> d;
	d.offset = this->desc_.start;
	d.offset += do_slice_dim<N>(this->desc_, d, slice{0});
	d.offset += do_slice_dim<N - 1>(this->desc_, d, slice{i, j - i + 1});
	
	std::size_t NRest = N - 2;
	while (NRest >= 1) {
	  d.start += do_slice_dim2(this->desc_, d, slice{0}, NRest);
	  --NRest;
	}
	return {d, data()};
}

template <typename T, std::size_t N>
template <typename F>
Tensor<T, N>& Tensor<T, N>::apply(F f) 
{
  for (auto &x : elements)
  {
  	  f(x);  
  }
  return *this;
}

template <typename T, std::size_t N>
template <typename M, typename F>
Enable_if<Tensor_type<M>(), Tensor<T, N> &> Tensor<T, N>::apply(const M &m, F f) 
{
  assert( equal_extent(this->descriptor, m.Descriptor()) );
  auto j = m.begin();
  for (auto i = begin(); i != end(); ++i) 
  {
    f(*i, *j);
    ++j;
  }
  return *this;
}


//uncomment this later

template < typename T, size_t N>
std::ostream& operator<<( std::ostream& os, Tensor<T,N>& m)
{
	os << '{';
	for ( size_t i = 0; i!= m.n_rows(); ++i )
	{
		os << m[i];
		if ( i + 1 != m.n_rows() )
		{
			os << ',';
		}
	}
	return os << '}';
}


/*
 * Tensor<T,N>& Tensor<T,N>::apply( F f ) takes a function object f and applies 
 * it to every element in the Tensor.
 */

template < typename T, size_t N >
template < typename F >
constexpr Tensor<T,N>& Tensor<T,N>::apply_to_each_element( F f) //applies a function to each element of the tensor
{
	for ( auto& x : elements ) 
	{
		f(x);
	}
	return *this;
}
//applies f(x,c) to each element of the tensor 
//x is each element in the tensor and c is some scalar value.
template< typename T, size_t N >
template < typename F >
constexpr Tensor<T,N>& Tensor<T,N>::apply_to_each_element( T& c ,F f)
{															 
	for ( auto& x : elements )
	{
		f(x,c);
	}
	return *this;
};



//to be uncommented
//applies a function to each element of a tensor 
template < typename T, size_t N >
template <typename F, typename M>
constexpr Tensor<T,N>& Tensor<T,N>::apply_to_each_element( M& m, F f )
{
	assert( equal_extent( this->descriptor, m.Descriptor() ) );
	for ( auto i = begin(), j = m.begin(); i != end(); ++i, ++j )
	{
		f(*i, *j);
	}
	return *this;
}

template <typename T, size_t N>
Tensor<T,N>& Tensor<T,N>::operator=( const T& scalar )
{
	return apply([&](T& a){ a = scalar; });
}

template < typename T, size_t N >
Tensor<T,N>& Tensor<T,N>::operator+=( const T& scalar ){ return apply_to_each_element( scalar, add_op_assign<T>() ); }

template < typename T, size_t N >
Tensor<T,N>& Tensor<T,N>::operator-=( const T& scalar ){ return apply_to_each_element( scalar, sub_op_assign<T>() ); }

template < typename T, size_t N >
Tensor<T,N>& Tensor<T,N>::operator*=( const T& scalar ){ return apply_to_each_element( scalar, mul_op_assign<T>() ); }

template < typename T, size_t N >
Tensor<T,N>& Tensor<T,N>::operator/=( const T& scalar ){ return apply_to_each_element( scalar, div_op_assign<T>() ); }

template < typename T, size_t N >
Tensor<T,N>& Tensor<T,N>::operator%=( const T& scalar ){ return apply_to_each_element( scalar, mod_op_assign<T>() ); }


template <typename T, size_t N>
Tensor<T,N>& Tensor<T,N>::operator+=( const Tensor<T,N>& tensor2 ){ return apply_to_each_element( tensor2, add_op_assign<T>() ); }

template <typename T, size_t N>
Tensor<T,N>& Tensor<T,N>::operator-=( const Tensor<T,N>& tensor2 ){ return apply_to_each_element( tensor2, sub_op_assign<T>() ); }

template <typename T, size_t N>
Tensor<T,N>& Tensor<T,N>::operator*=( const Tensor<T,N>& tensor2 ){ return apply_to_each_element( tensor2, mul_op_assign<T>() ); }

template <typename T, size_t N>
Tensor<T,N>& Tensor<T,N>::operator/=( const Tensor<T,N>& tensor2 ){ return apply_to_each_element( tensor2, div_op_assign<T>() ); }

template <typename T, size_t N>
Tensor<T,N>& Tensor<T,N>::operator%=( const Tensor<T,N>& tensor2 ){ return apply_to_each_element( tensor2, mod_op_assign<T>() ); }


template <typename T, std::size_t N>
template <typename M>
Enable_if< Tensor_type<M>(), Tensor<T, N>& > Tensor<T, N>::operator+=( const M &m) 
{
	assert( equal_extent( this->descriptor, m.Descriptor()) );  
	return apply(m, [&](T &a, const Value_type<M> &b) { a += b; });
}

template <typename T, std::size_t N>
template <typename M>
Enable_if< Tensor_type<M>(), Tensor<T, N> &> Tensor<T, N>::operator-=( const M &m) 
{
	assert( equal_extent(this->descriptor, m.Descriptor()) );  
	return apply(m, [&](T &a, const Value_type<M> &b) { a -= b; });
}

template <typename T, std::size_t N>
template <typename M>
Enable_if<Tensor_type<M>(), Tensor<T, N> &> Tensor<T, N>::operator*=( const M &m) 
{
  assert( equal_extent( this->descriptor, m.Descriptor() ) );  
  return apply(m, [&](T &a, const Value_type<M> &b) { a *= b; });
}

template <typename T, std::size_t N>
template <typename M>
Enable_if<Tensor_type<M>(), Tensor<T, N> &> Tensor<T, N>::operator/=( const M &m) 
{
  assert( equal_extent( this->descriptor, m.Descriptor() ) );  
  return apply(m, [&](T &a, const Value_type<M> &b) { a /= b; });
}

template <typename T, std::size_t N>
template <typename M>
Enable_if<Tensor_type<M>(), Tensor<T, N> &> Tensor<T, N>::operator%=( const M &m) 
{
  assert( equal_extent( this->descriptor, m.Descriptor() ) );  
  return apply(m, [&](T &a, const Value_type<M> &b) { a %= b; });
}


template <typename T, size_t N>
template <typename U>
Tensor<U,N> Tensor<T,N>::operator-() const
{
	Tensor<U,N> res(*this); //deep copy copy constructor gets invoked
	return res.apply_to_each_element( negate_all<T>() );
}


template <typename T, size_t N>
void Tensor<T,N>::clear()
{
	this->descriptor.clear();
	elements.clear();
}


/*---------------------------------------x----------------------------------*/
/*The following are the binary arithmetic operations of a Tensor<T,N> with another
 *Tensor<T,N>*/
/*---------------------------------------x----------------------------------*/

template <typename T1, typename T2>
inline 
Enable_if< Tensor_type<T1>() && Tensor_type<T2>(), bool >
operator==( const T1& a, const T2& b )
{
	assert(equal_extent(a.Descriptor(),b.Descriptor()));
	return std::equal( a.begin(), a.end(), b.begin());
}


template <typename T1, typename T2>
inline 
Enable_if< Tensor_type<T1>() && Tensor_type<T2>(), bool >
operator!=( const T1& a, const T2& b )
{
	return !( a == b );
}

template <typename T, std::size_t N>
Tensor<T, N> operator+(const Tensor<T, N> &x, const T &val) 
{
	Tensor<T, N> res = x;
	res += val;
	return res;
}


template <typename T, std::size_t N>
Tensor<T, N> operator+(const TensorRef<T, N> &x, const T &val) 
{
	Tensor<T, N> res = x;
	res += val;
	return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator+(const T& val, const Tensor<T, N> &x )
{
	Tensor<T, N> res = x;
	res += val;
	return res;
}


template <typename T, std::size_t N>
Tensor<T, N> operator+(const T& val, const TensorRef<T, N> &x ) 
{
	Tensor<T, N> res = x;
	res += val;
	return res;
}



template <typename T, std::size_t N>
Tensor<T, N> operator-(const Tensor<T, N> &x, const T &val) 
{
	Tensor<T, N> res = x;
	res -= val;
	return res;
}


template <typename T, std::size_t N>
Tensor<T, N> operator-(const TensorRef<T, N> &x, const T &val) 
{
	Tensor<T, N> res = x;
	res -= val;
	return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator*(const Tensor<T, N> &x, const T &val) 
{
	Tensor<T, N> res = x;
	res *= val;
	return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator*(const TensorRef<T, N> &x, const T &val) 
{
	Tensor<T, N> res = x;
	res *= val;
	return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator*(const T &val, const Tensor<T, N> &x) 
{
	Tensor<T, N> res = x;
	res *= val;
	return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator*(const T &val, const TensorRef<T, N> &x) 
{
	Tensor<T, N> res = x;
	res *= val;
	return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator/(const Tensor<T, N> &x, const T &val) 
{
  Tensor<T, N> res = x;
  res /= val;
  return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator/(const TensorRef<T, N> &x, const T &val) 
{
	Tensor<T, N> res = x;
	res /= val;
	return res;
}


template <typename T, std::size_t N>
Tensor<T, N> operator%(const Tensor<T, N> &x, const T &val) 
{
	Tensor<T, N> res = x;
	res %= val;
	return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator%(const TensorRef<T, N> &x, const T &val) 
{
	Tensor<T, N> res = x;
	res %= val;
	return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator+(const Tensor<T, N> &a, const Tensor<T, N> &b) 
{
	Tensor<T, N> res = a;
	res += b;
	return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator+(const TensorRef<T, N> &a, const TensorRef<T, N> &b) 
{
	Tensor<T, N> res = a;
	res += b;
	return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator+(const Tensor<T, N> &a, const TensorRef<T, N> &b) 
{
	Tensor<T, N> res = a;
	res += b;
	return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator+(const TensorRef<T, N> &a, const Tensor<T, N> &b) 
{
	Tensor<T, N> res = a;
	res += b;
	return res;
}


template <typename T, std::size_t N>
Tensor<T, N> operator-(const Tensor<T, N> &a, const Tensor<T, N> &b) 
{
	Tensor<T, N> res = a;
	res -= b;
	return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator-(const TensorRef<T, N> &a, const TensorRef<T, N> &b) 
{
	Tensor<T, N> res = a;
	res -= b;
	return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator-(const Tensor<T, N> &a, const TensorRef<T, N> &b) 
{
	Tensor<T, N> res = a;
	res -= b;
	return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator-(const TensorRef<T, N> &a, const Tensor<T, N> &b) 
{
	Tensor<T, N> res = a;
	res -= b;
	return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator*(const Tensor<T, N> &a, const Tensor<T, N> &b) 
{
	Tensor<T, N> res = a;
	res *= b;
	return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator*(const TensorRef<T, N> &a, const TensorRef<T, N> &b) 
{
	Tensor<T, N> res = a;
	res *= b;
	return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator*(const Tensor<T, N> &a, const TensorRef<T, N> &b) 
{
	Tensor<T, N> res = a;
	res *= b;
	return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator*(const TensorRef<T, N> &a, const Tensor<T, N> &b) 
{
	Tensor<T, N> res = a;
	res *= b;
	return res;
}


template <typename T, std::size_t N>
Tensor<T, N> operator/(const Tensor<T, N> &a, const Tensor<T, N> &b) 
{
	Tensor<T, N> res = a;
	res /= b;
	return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator/(const TensorRef<T, N> &a, const TensorRef<T, N> &b) 
{
	Tensor<T, N> res = a;
	res /= b;
	return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator/(const Tensor<T, N> &a, const TensorRef<T, N> &b) 
{
	Tensor<T, N> res = a;
	res /= b;
	return res;
}

template <typename T, std::size_t N>
Tensor<T, N> operator/(const TensorRef<T, N> &a, const Tensor<T, N> &b) 
{
	Tensor<T, N> res = a;
	res /= b;
	return res;
}







/*---------------------------------------x----------------------------------*/




template <typename T, size_t N>
template <typename U, std::size_t NN, typename X>
Tensor<T,N>::Tensor(const Tensor<U, 2> &x)
	:TensorBase<T,N>{ x.n_rows() }
	,elements{x.begin(), x.end()}
{
	static_assert(Convertible<U,T>(), "Type incompatibility error.");
	assert(x.n_cols() == 1);
}


template <typename T, size_t N>
template <typename U, std::size_t NN, typename X>
Tensor<T,N>::Tensor(const TensorRef<U, 2> &x)
	:TensorBase<T,N>{ x.n_rows() }
	,elements{x.begin(), x.end()}
{
	static_assert(Convertible<U,T>(), "Type incompatibility error.");
	assert(x.n_cols() == 1);
}



template <typename T, size_t N>
template <typename U, std::size_t NN, typename X>
Tensor<T,N>& Tensor<T,N>::operator=(const Tensor<U, 2> &x)
{
	static_assert(Convertible<U, T>(), "Tensor =: incompatible element types");
	assert(x.n_cols() == 1);

	this->descriptor.size = x.Descriptor().size;
	this->descriptor.offset = 0;
	this->descriptor.extents[0] = x.n_rows();
	this->descriptor.strides[0] = 1;
	elements.assign(x.begin(), x.end());
	return *this;
}

template <typename T, size_t N>
template <typename U, std::size_t NN, typename X>
Tensor<T,N>& Tensor<T,N>::operator=(const TensorRef<U, 2> &x)
{
	static_assert(Convertible<U, T>(), "Tensor =: incompatible element types");
	assert(x.n_cols() == 1);

	this->descriptor.size = x.Descriptor().size;
	this->descriptor.offset = 0;
	this->descriptor.extents[0] = x.n_rows();
	this->descriptor.strides[0] = 1;
	elements.assign(x.begin(), x.end());
	return *this;
}




//matrix from vector
template <typename T, size_t N>
template <typename U, std::size_t NN, typename X>
Tensor<T,N>::Tensor(const Tensor<U, 1> &x)
	:TensorBase<T,N>{x.n_rows(), 1}
	,elements{x.begin(), x.end()}
{
	static_assert(Convertible<U,T>(), "Incompatible type error");
}



template <typename T, size_t N>
template <typename U, std::size_t NN, typename X>
Tensor<T,N>::Tensor(const TensorRef<U, 1> &x)
	:TensorBase<T,N>{x.n_rows(), 1}
	,elements{x.begin(), x.end()}
{
	static_assert(Convertible<U,T>(), "Incompatible type error");
}


template <typename T, size_t N>
template <typename U, std::size_t NN, typename X>
Tensor<T,N>& Tensor<T,N>::operator=(const Tensor<U, 1> &x)
{
	static_assert(Convertible<U, T>(), "Tensor = Incompatible Element Types");
	
	this->descriptor.size = x.Descriptor().size;
	this->descriptor.offset = 0;
	this->descriptor.extents[0] = x.n_rows();
	this->descriptor.extents[1] = 1;
	this->descriptor.strides[0] = x.n_rows();
	this->descriptor.strides[1] = 1;
	elements.assign(x.begin(), x.end());
	return *this;
}

template <typename T, size_t N>
template <typename U, std::size_t NN, typename X>
Tensor<T,N>& Tensor<T,N>::operator=(const TensorRef<U, 1> &x)
{
	static_assert(Convertible<U, T>(), "Tensor = Incompatible Element Types");
	
	this->descriptor.size = x.Descriptor().size;
	this->descriptor.offset = 0;
	this->descriptor.extents[0] = x.n_rows();
	this->descriptor.extents[1] = 1;
	this->descriptor.strides[0] = x.n_rows();
	this->descriptor.strides[1] = 1;
	elements.assign(x.begin(), x.end());
	return *this;
}









template <typename T>
class Tensor<T,0> : public TensorBase<T,0>
{
public:
	using iterator = typename std::array<T,1>::iterator;
	using const_iterator = typename std::array<T,1>::const_iterator;

	Tensor( const T& x = T{} )
		:elements{x}
	{}

	Tensor& operator=( const T& x )
	{
		elements[0] = x;
		return *this;
	}

	size_t size()
	{
		return 1;
	}

	T* data() { return elements.data(); }
	const T* data() const { return elements.data(); }

	operator T&() { return elements[0]; }
	operator const T&() { return elements[0]; }

	iterator begin() { return elements.begin(); }
	const_iterator begin() const { return elements.cbegin(); }
	iterator end() { return elements.end(); }
	const_iterator end() const { return elements.cend(); }


protected:
	std::array<T,1> elements;
};

template <typename T>
std::ostream& operator<<( std::ostream& os, const Tensor<T,0>& m )
{
	return os << (const T&)m();
}




} //namespace linalg



#endif
