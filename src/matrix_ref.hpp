#ifndef MATRIX_REF_H
#define MATRIX_REF_H

#include <iostream>
#include <vector>
#include "matrix_slice.hpp"
//#include "matrix_base.hpp"
#include "matrix_initializer.hpp"
#include "matrix_base_base.hpp"
namespace linalg
{

template <typename T, size_t N>
class TensorRefIter;


template < typename T, size_t N >
class TensorRef : public TensorBase<T,N>
{
public:
	using iterator = TensorRefIter<T,N>;
	using const_iterator = TensorRefIter< const T,N>;
	
	TensorRef()                               = delete;
	TensorRef( TensorRef&& )                  = default; //move
	TensorRef( const TensorRef& )             = default; //copy
	TensorRef& operator=(TensorRef&&);         //move
	TensorRef& operator=( const TensorRef& );  //copy
	~TensorRef()  						      = default;
	
	template <typename U>
	TensorRef( const TensorRef<U,N>& x );

	template <typename U>
	TensorRef& operator=( const TensorRef<U,N>& x );

	template< typename U>
	TensorRef( const Tensor<U,N>& x );

	template <typename U>
	TensorRef& operator=( const Tensor<U,N>& x );

	TensorRef& operator=( TensorInitializer<T,N> );
	
	TensorRef( const TensorSlice<N>& s, T* p)
		:TensorBase<T,N>{s}
		,elements{p}
	{}

	size_t size() { return this->descriptor.size; }

	T* data() { return elements; }
	const T* data() const { return elements; }

	using TensorBase<T,N>::operator();

	template <typename... Args>
	Enable_if<RequestingSlice<Args...>(), TensorRef<T, N>>
	operator()(const Args &... args);

	template <typename... Args>
	Enable_if<RequestingSlice<Args...>(), TensorRef<const T, N>>
	operator()(const Args &... args) const;

	TensorRef<T, N - 1> operator[](size_t i) { return row(i); }
	TensorRef<const T, N - 1> operator[](size_t i) const { return row(i); }

	TensorRef<T,N-1> row( size_t n );
	TensorRef<const T, N-1> row( size_t n) const;

	TensorRef<T, N - 1> col(size_t n);
	TensorRef<const T, N - 1> col(size_t n) const;
	TensorRef<T, N> rows(size_t i, size_t j);
	TensorRef<const T, N> rows(size_t i, size_t j) const;
	TensorRef<T, N> cols(size_t i, size_t j);
	TensorRef<const T, N> cols(size_t i, size_t j) const;

	iterator begin() { return (this->descriptor, elements); }
	const_iterator begin() const { return (this->descriptor, elements); }
	iterator end() { return (this->descriptor, elements, true); }
	const_iterator end() const { return (this->descriptor, elements, true); }


	template <typename F>
	TensorRef &apply(F f);  // f(x) for every element x
	
	// f(x, mx) for corresponding elements of *this and m
	template <typename M, typename F>
	Enable_if<Tensor_type<M>(), TensorRef&> apply(const M &m, F f);
	
	 TensorRef&operator= (const T &value);   // assignment with scalar
	 TensorRef&operator+=(const T &value);  // scalar addition
	 TensorRef&operator-=(const T &value);  // scalar subtraction
	 TensorRef&operator*=(const T &value);  // scalar multiplication
	 TensorRef&operator/=(const T &value);  // scalar division
	 TensorRef&operator%=(const T &value);  // scalar modulo
	
	template <typename M>
	Enable_if<Tensor_type<M>(), TensorRef&> operator+=(const M &x);

	template <typename M>
	Enable_if<Tensor_type<M>(), TensorRef&> operator-=(const M &x);

	template <typename M>
	Enable_if<Tensor_type<M>(), TensorRef&> operator*=(const M &x);

	template <typename M>
	Enable_if<Tensor_type<M>(), TensorRef&> operator/=(const M &x);

	template <typename M>
	Enable_if<Tensor_type<M>(), TensorRef&> operator%=(const M &x);
	
	template <typename U = typename std::remove_const<T>::type>
	Tensor<U, N> operator-() const;
	
	//vector from tensor 
	template <typename U, size_t NN = N, typename = Enable_if<(NN == 1)>>
	TensorRef &operator=(const TensorRef<U, 2> &x);
	template <typename U, size_t NN = N, typename = Enable_if<(NN == 1)>>
	TensorRef &operator=(const Tensor<U, 2> &x);

	template <size_t NN = N, typename = Enable_if<(NN == 1)>>
	std::vector<T> GetTostdVec() const 
	{
		return std::vector<T>(begin(), end());
	}
  
  	//tensor from vector
	template <typename U, size_t NN = N, typename = Enable_if<(NN == 2)>>
	TensorRef &operator=(const TensorRef<U, 1> &x);
	template <typename U, size_t NN = N, typename = Enable_if<(NN == 2)>>
	TensorRef &operator=(const Tensor<U, 1> &x);


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


	template < size_t NN = N, typename = Enable_if<(NN==1) || (NN==2)> >
	Tensor<T,2> transpose() const { return Transpose( *this ); }



protected:
	T* elements;

};


template <typename T, size_t N>
TensorRef<T,N>& TensorRef<T,N>::operator=( TensorRef&& m ) //move
{
	assert( equal_extent(this->descriptor, m.Descriptor()));
	std::move(m.begin(), m.end(), begin());
	return *this;
}


template <typename T, size_t N>
TensorRef<T,N>& TensorRef<T,N>::operator=( const TensorRef& m ) //copy
{
	assert( equal_extent(this->descriptor, m.Descriptor()));
	std::copy(m.begin(), m.end(), begin());
	return *this;
}

template <typename T, size_t N>
template< typename U>
TensorRef<T,N>::TensorRef( const TensorRef<U,N>& x )
	:TensorBase<T,N>{x.Descriptor()}
	,elements{x.data()}
{}

template <typename T, size_t N>
template <typename U>
TensorRef<T, N> &TensorRef<T, N>::operator=(const TensorRef<U, N> &x) 
{
	static_assert(Convertible<U, T>(), "TensorRef =: incompatible element types");
	assert(this->descriptor.extents == x.Descriptor().extents);	
	std::copy(x.begin(), x.end(), begin());
	return *this;
}

template <typename T, size_t N>
template <typename U>
TensorRef<T, N>::TensorRef(const Tensor<U, N> &x)
    : TensorBase<T, N>{x.Descriptor()}
    , elements(x.data()) 
{}



template <typename T, std::size_t N>
template <typename U>
TensorRef<T, N> &TensorRef<T, N>::operator=(const Tensor<U, N> &x) 
{
	static_assert(Convertible<U, T>(), "TensorRef =: incompatible element types");
	assert(this->descriptor.extents == x.Descriptor().extents);
	
	std::copy(x.begin(), x.end(), begin());
	return *this;
}

template <typename T, std::size_t N>
TensorRef<T, N> &TensorRef<T, N>::operator=(TensorInitializer<T, N> init) 
{
	assert(derive_extents<N>(init) == this->descriptor.extents);
	auto iter = begin();
	copy_flat(init, iter);
	return *this;
}


template <typename T, size_t N>
template <typename... Args>
Enable_if<RequestingSlice<Args...>(), TensorRef<T, N>>
TensorRef<T, N>::operator()(const Args &... args) 
{
	TensorSlice<N> d;
	d.offset = this->descriptor.offset +do_slice(this->descriptor, d, args...);
	d.size = compute_size(d.extents);
	return (d, data());
}

template <typename T, size_t N>
template <typename... Args>
Enable_if<RequestingSlice<Args...>(), TensorRef<const T, N>>
TensorRef<T, N>::operator()(const Args &... args) const 
{
	TensorSlice<N> d;
	d.offset = this->descriptor.offset + do_slice(this->descriptor, d, args...);
	d.size = compute_size(d.extents);
	return (d, data());
}

template <typename T, size_t N>
TensorRef<T, N - 1> TensorRef<T, N>::row(size_t n) 
{
	assert(n < this->n_rows());
	TensorSlice<N - 1> row;
	slice_dim<0>(n, this->descriptor, row);
	return (row, elements);
}

template <typename T, size_t N>
TensorRef<const T, N - 1> TensorRef<T, N>::row(size_t n) const 
{
	assert(n < this->n_rows());
	TensorSlice<N - 1> row;
	slice_dim<0>(n, this->descriptor, row);
	return (row, elements);
}

// col
template <typename T, size_t N>
TensorRef<T, N - 1> TensorRef<T, N>::col(size_t n) 
{
	assert(n < this->n_cols());
	TensorSlice<N - 1> col;
	slice_dim<1>(n, this->descriptor, col);
	return (col, elements);
}

template <typename T, size_t N>
TensorRef<const T, N - 1> TensorRef<T, N>::col(size_t n) const 
{
	assert(n < this->n_cols());
	TensorSlice<N - 1> col;
	slice_dim<1>(n, this->descriptor, col);
	return (col, elements);
}

template <typename T, std::size_t N>
TensorRef<T, N> TensorRef<T, N>::rows(std::size_t i, std::size_t j) 
{
	assert(i < j);
	assert(j < this->n_rows());
	
	TensorSlice<N> d;
	d.offset = this->descriptor.offset;
	d.offset += do_slice_dim<N>(this->descriptor, d, slice{i, j - i + 1});
	std::size_t NRest = N - 1;
	while (NRest >= 1) 
	{
		d.offset += do_slice_dim2(this->descriptor, d, slice{0}, NRest);
		--NRest;
	}
	return {d, data()};
}

template <typename T, std::size_t N>
TensorRef<const T, N> TensorRef<T, N>::rows(std::size_t i, std::size_t j) const 
{
	assert(i < j);
	assert(j < this->n_rows());
	
	TensorSlice<N> d;
	d.offset = this->descriptor.offset;
	d.offset += do_slice_dim<N>(this->descriptor, d, slice{i, j - i + 1});
	std::size_t NRest = N - 1;
	while (NRest >= 1) 
	{
		d.offset += do_slice_dim2(this->descriptor, d, slice{0}, NRest);
		--NRest;
	}
	return {d, data()};
}



template <typename T, std::size_t N>
TensorRef<T, N> TensorRef<T, N>::cols(std::size_t i, std::size_t j) 
{
  assert(N >= 2);
  assert(i < j);
  assert(j < this->n_cols());

  TensorSlice<N> d;
  d.offset = this->descriptor.offset;
  d.offset += do_slice_dim<N>(this->descriptor, d, slice{0});
  d.offset += do_slice_dim<N - 1>(this->desc_, d, slice{i, j - i + 1});

  std::size_t NRest = N - 2;
  while (NRest >= 1) 
  {
    d.offset += do_slice_dim2(this->descriptor, d, slice{0}, NRest);
    --NRest;
  }
  return {d, data()};
}


template <typename T, std::size_t N>
TensorRef< const T, N> TensorRef<T, N>::cols(std::size_t i, std::size_t j) const  
{
  assert(N >= 2);
  assert(i < j);
  assert(j < this->n_cols());

  TensorSlice<N> d;
  d.offset = this->descriptor.offset;
  d.offset += do_slice_dim<N>(this->descriptor, d, slice{0});
  d.offset += do_slice_dim<N - 1>(this->desc_, d, slice{i, j - i + 1});

  std::size_t NRest = N - 2;
  while (NRest >= 1) 
  {
    d.offset += do_slice_dim2(this->descriptor, d, slice{0}, NRest);
    --NRest;
  }
  return {d, data()};
}



template <typename T, std::size_t N>
template <typename F>
TensorRef<T, N>& TensorRef<T, N>::apply(F f) 
{
  for (auto &x : elements)
  {
  	  f(x);  
  }
  return *this;
}

template <typename T, std::size_t N>
template <typename M, typename F>
Enable_if<Tensor_type<M>(), TensorRef<T, N> &> TensorRef<T, N>::apply(const M &m, F f) 
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
template <typename T, std::size_t N>
TensorRef<T, N> &TensorRef<T, N>::operator=(const T &val) {
  return apply([&](T &a) { a = val; });
}

template <typename T, std::size_t N>
TensorRef<T, N> &TensorRef<T, N>::operator+=(const T &val) {
  return apply([&](T &a) { a += val; });
}

template <typename T, std::size_t N>
TensorRef<T, N> &TensorRef<T, N>::operator-=(const T &val) {
  return apply([&](T &a) { a -= val; });
}

template <typename T, std::size_t N>
TensorRef<T, N> &TensorRef<T, N>::operator*=(const T &val) {
  return apply([&](T &a) { a *= val; });
}

template <typename T, std::size_t N>
TensorRef<T, N> &TensorRef<T, N>::operator/=(const T &val) {
  return apply([&](T &a) { a /= val; });
}

template <typename T, std::size_t N>
TensorRef<T, N> &TensorRef<T, N>::operator%=(const T &val) {
  return apply([&](T &a) { a %= val; });
}

template <typename T, std::size_t N>
template <typename M>
Enable_if<Tensor_type<M>(), TensorRef<T, N> &> TensorRef<T, N>::operator+=(const M &m) 
{
  assert(equal_extent(this->descriptor, m.Descriptor()));  
  return apply(m, [&](T &a, const Value_type<M> &b) { a += b; });
}

template <typename T, std::size_t N>
template <typename M>
Enable_if<Tensor_type<M>(), TensorRef<T, N> &> TensorRef<T, N>::operator-=( const M &m) 
{
  assert(equal_extent(this->descriptor, m.Descriptor()));  
  return apply(m, [&](T &a, const Value_type<M> &b) { a -= b; });
}

template <typename T, std::size_t N>
template <typename M>
Enable_if<Tensor_type<M>(), TensorRef<T, N> &> TensorRef<T, N>::operator*=(const M &m) 
{
  assert(equal_extent(this->descriptor, m.Descriptor()));  
  return apply(m, [&](T &a, const Value_type<M> &b) { a *= b; });
}

template <typename T, std::size_t N>
template <typename M>
Enable_if<Tensor_type<M>(), TensorRef<T, N> &> TensorRef<T, N>::operator/=(const M &m) 
{
  assert(equal_extent(this->descriptor, m.Descriptor()));  
  return apply(m, [&](T &a, const Value_type<M> &b) { a /= b; });
}

template <typename T, std::size_t N>
template <typename M>
Enable_if<Tensor_type<M>(), TensorRef<T, N> &> TensorRef<T, N>::operator%=(const M &m) 
{
  assert(equal_extent(this->descriptor, m.Descriptor()));  
  return apply(m, [&](T &a, const Value_type<M> &b) { a %= b; });
}

template <typename T, std::size_t N>
template <typename U>
Tensor<U, N> TensorRef<T, N>::operator-() const 
{
  Tensor<U, N> res(*this);
  return res.apply([&](U &a) { a = -a; });
}

template <typename T, std::size_t N>
template <typename U, std::size_t NN, typename X>
TensorRef<T, N> &TensorRef<T, N>::operator=(const TensorRef<U, 2> &x) {
  static_assert(Convertible<U, T>(), "TensorRef = incompatible element types");
  assert(this->size() == x.size());
  assert(x.n_cols() == 1);
  std::copy(x.begin(), x.end(), begin());
  return *this;
}

template <typename T, std::size_t N>
template <typename U, std::size_t NN, typename X>
TensorRef<T, N> &TensorRef<T, N>::operator=(const Tensor<U, 2> &x) 
{
  static_assert(Convertible<U, T>(), "Tensor =: incompatible element types");
  assert(this->size() == x.size());
  assert(x.n_cols() == 1);
  std::copy(x.begin(), x.end(), begin());
  return *this;
}

template <typename T, std::size_t N>
template <typename U, std::size_t NN, typename X>
TensorRef<T, N> &TensorRef<T, N>::operator=(const TensorRef<U, 1> &x) 
{
  static_assert(Convertible<U, T>(), "TensorRef =: incompatible element types");
  assert(this->size() == x.size());
  assert(this->n_cols() == 1);
  std::copy(x.begin(), x.end(), begin());
  return *this;
}

template <typename T, std::size_t N>
template <typename U, std::size_t NN, typename X>
TensorRef<T, N> &TensorRef<T, N>::operator=(const Tensor<U, 1> &x) 
{
  static_assert(Convertible<U, T>(), "TensorRef =: incompatible element types");
  assert(this->size() == x.size());
  assert(this->n_cols() == 1);
  std::copy(x.begin(), x.end(), begin());
  return *this;
}


template <typename T>
class TensorRef<T, 0> : public TensorBase<T, 0> 
{ 
public:
  using iterator = T *;
  using const_iterator = const T *;

  TensorRef(const TensorSlice<0> &s, T *p) : elements{p + s.offset} {}

  size_t size() { return 1; }

  T *data() { return elements; }
  const T *data() const { return elements; }

  T &operator()() { return *elements; };
  const T &operator()() const { return *elements; }

  operator T &() { return *elements; }
  operator const T &() const { return *elements; }

  iterator begin() { return iterator(data()); }
  const_iterator begin() const { return const_iterator(data()); }
  iterator end() { return iterator(data() + 1); }
  const_iterator end() const { return const_iterator(data() + 1); }

protected:
  T* elements;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const TensorRef<T, 0> &mr0) {
  return os << (const T &)mr0;
}



template <typename T, size_t N>
class TensorRefIter 
{
  template <typename U, size_t NN>
  friend std::ostream &operator<<(std::ostream &os,
                                  const TensorRefIter<U, NN> &iter);

 public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = typename std::remove_const<T>::type;
  using pointer = T *;
  using reference = T &;
  using difference_type = std::ptrdiff_t;

  TensorRefIter(const TensorSlice<N> &s, T *base, bool limit = false);
  TensorRefIter &operator=(const TensorRefIter &);

  const TensorSlice<N>& Descriptor() const { return descriptor; }

  T &operator*() { return *elements; }
  T *operator->() { return elements; }

  const T& operator*() const { return *elements; }
  const T* operator->() const { return elements; }

  TensorRefIter &operator++();
  TensorRefIter operator++(int);

 private:
  void increment();

  std::array<size_t, N> index;
  const TensorSlice<N>& descriptor;
  T *elements;
};

template <typename T, std::size_t N>
TensorRefIter<T, N>::TensorRefIter(const TensorSlice<N> &s, T *base,
                                           bool limit)
    : descriptor(s) 
{
  std::fill(index.begin(), index.end(), 0);

  if (limit) 
  {
    index[0] = descriptor.extents[0];
    elements = base + descriptor.Offset(index);
  } 
  else 
  {
    elements = base + s.offset;
  }
}

template <typename T, std::size_t N>
TensorRefIter<T, N> &TensorRefIter<T, N>::operator=( const TensorRefIter &iter) 
{
  std::copy(iter.index.begin(), iter.index.end(), index.begin());
  elements = iter.elements;
  return *this;
}

template <typename T, std::size_t N>
TensorRefIter<T, N> &TensorRefIter<T, N>::operator++() 
{
  increment();
  return *this;
}

template <typename T, std::size_t N>
TensorRefIter<T, N> TensorRefIter<T, N>::operator++(int) 
{
  TensorRefIter<T, N> x = *this;
  increment();
  return *x;
}

template <typename T, std::size_t N>
void TensorRefIter<T, N>::increment() 
{
  std::size_t d = N - 1;

  while (true) {
    elements += descriptor.strides[d];
    ++index[d];

    if (index[d] != descriptor.extents[d]) break;

    if (d != 0) {
      elements -= descriptor.strides[d] * descriptor.extents[d];
      index[d] = 0;
      --d;
    } else {
      break;
    }
  }
}

template <typename T, size_t N>
std::ostream &operator<<(std::ostream &os,
                         const TensorRefIter<T, N> &iter) {
  os << "target: " << *iter.elements << ", indx: " << iter.index << std::endl;
  return os;
}

template <typename T, std::size_t N>
inline bool operator==(const TensorRefIter<T, N> &a,
                       const TensorRefIter<T, N> &b) 
{
  assert(a.Descriptor() == b.Descriptor());
  return &*a == &*b;
}

template <typename T, std::size_t N>
inline bool operator!=(const TensorRefIter<T, N> &a,
                       const TensorRefIter<T, N> &b) 
{
  return !(a == b);
}







}//namespace linalg
#endif

