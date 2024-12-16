#ifndef LINALG_H
#define LINALG_H

//#include <initializer_list>
//#include <iterator>
//#include <exception>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>


#include "exceptions.hpp"
#include <initializer_list>
using std::initializer_list;
/*  Implementing a custom vector class and a matrix class and defining methods
 *   on them for basic linear algebra
*/
namespace linalg
{

template < typename ValType >
class Array1d
{
public:
	virtual size_t getdims() const = 0;
	virtual void Show() const = 0;
};


template < typename T >
class Vector : public Array1d<T>
{
public:
	using ValType = T;
	//following two lines to initialize
	//template <typename... kwargs> Vector( kwargs... inputs );
	


	Vector();							//defined
	Vector( unsigned int user_size, ValType user_val);
	Vector( initializer_list<T> args );
	Vector( const ValType* initialindex, const ValType* finalindex );
	Vector( const Vector& othervec ); //copy constructor
	Vector<ValType>& operator=(const Vector& copysource);      //copy assignment operator
	Vector( Vector&& othervec ) noexcept;					//move constructor
	Vector<ValType>& operator=( Vector&& movesource ) noexcept; //move assignment operator
	virtual ~Vector();								//destructor
	
	//api / feataures
	virtual void push_back( const ValType& insert_element );		//defined
	virtual size_t Capacity() const;                                //defined
	virtual size_t size() const; 									//defined
	ValType getindex( unsigned int n ) const;  //defined
	void setindex(const unsigned int& index, const ValType& valtobeput ) const; //defined
	size_t getdims() const;
	void Show() const;
	virtual double Norm();
	//virtual double length();
	virtual double lpnorm( double p );
	virtual ValType* begin() const ;
	virtual ValType* end() const ;
	virtual std::vector< ValType >& GetTostdVec( std::vector< ValType >& temp ) const;
	virtual Vector< ValType > cross(const Vector< ValType >& vec2) const;
	void realloc( size_t newcapacity ); 

	T operator[]( size_t n ) { return vec[n]; }
	const T operator[]( size_t n ) const { return vec[n]; }
	

/*protected: //we do not need this stupid protect
our vector has only three fields, one to actually point to the 
memory block holding the data
the other to keep the overall required capacity in check
and the len to hold the actual current length
*/
	//data members
	size_t capacity{0}; 		
	size_t  len_{0};  	 	
	ValType *vec{nullptr};	 	
};

//proxy norm objects for faster norm evaluation of 2 and 3 element vectors.

class NormProxy2 
{
public:
	NormProxy2( double x, double y ) : squared_{x*x + y*y} {}
	bool operator==( const NormProxy2& other ) const 
	{
		return squared_ == other.squared_;
	}
	bool operator<( const NormProxy2& other ) const 
	{
		return squared_ < other.squared_;
	}
	bool operator>( const NormProxy2& other ) const 
	{
		return squared_ > other.squared_;
	}
	friend bool operator <( const NormProxy2& proxyobj, double len )
	{
		return proxyobj.squared_ < len*len;
	}
	friend bool operator >( const NormProxy2& proxyobj, double len )
	{
		return proxyobj.squared_ > len*len;
	}
	friend bool operator == ( const NormProxy2& proxyobj, double len )
	{
		return proxyobj.squared_ == len*len;
	}
	operator double() const &&
	{
			return std::sqrt(squared_);
	}
	operator float() const &&
	{
			return std::sqrt(squared_);
	}

protected:
	double squared_{};
};



class NormProxy3 
{
public:
	NormProxy3( double x, double y, double z) : squared_{x*x + y*y + z*z} {}
	bool operator==( const NormProxy3& other ) const 
	{
		return squared_ == other.squared_;
	}
	bool operator<( const NormProxy3& other ) const 
	{
		return squared_ < other.squared_;
	}
	bool operator>( const NormProxy3& other ) const 
	{
		return squared_ > other.squared_;
	}
	friend bool operator <( const NormProxy3& proxyobj, double len ) 
	{
		return proxyobj.squared_ < len*len;
	}
	friend bool operator >( const NormProxy3& proxyobj, double len ) 
	{
		return proxyobj.squared_ > len*len;
	}
	friend bool operator == ( const NormProxy3& proxyobj, double len )
	{
		return proxyobj.squared_ == len*len;
	}
	operator double() const  &&
	{
			return std::sqrt(squared_);
	}
	operator float() const  &&
	{
			return std::sqrt(squared_);
	}
protected:
	double squared_{};
};



template < typename T >
inline size_t Vector<T>::getdims() const
{
	return 1;
}

template < typename ValType >
inline size_t Vector< ValType >::Capacity() const
 {
 	 return capacity;
 }
//gives us the capacity of our vector

template < typename ValType >
inline size_t Vector< ValType >::size() const
{
	return len_;
}
//returns us the current length of our vector 


/*template < typename ValType >
inline unsigned int Vector< ValType >::isfine( unsigned int i )  
{
	static_assert( i > 0 && i < len_ , "OutOfBoundsError");
	return i;
}
to check if the index that the user is trying to access is even available or not
*/

template < typename ValType >
void Vector< ValType >::realloc( size_t newcapacity )
{
	//need an assert here	
	if ( newcapacity <= 0 )
	{
		throw( NullSizeError("The new capacity ought to be higher than zero.") );
	}
	else
	{
		ValType* temp = ( ValType* ) new ValType [newcapacity];
		
		if ( newcapacity < len_  )
		{
			len_ = newcapacity;
		}

		for( int i = 0; i < len_; i++)
		{
			*(temp + i) = std::move(*( vec+i ));
		}
		delete[] vec;
		vec = temp;
		capacity = newcapacity;
	}
}
//default constructor
template < typename ValType >
inline Vector< ValType >::Vector()
{	
	realloc(4);
}
//push_back function
template < typename ValType >
void Vector<ValType>::push_back(const ValType& insert_element )
{	
	if ( len_ >= capacity )
	{
		realloc( capacity + capacity/2 );
	}
	vec[len_] = std::move( insert_element );
	len_ += 1;	
}

//copy constructor
template < typename ValType >
Vector< ValType >::Vector(const Vector& othervec )
	:len_{othervec.len_}, capacity{othervec.capacity}
{
	//std::cout << "copy constructor invoked" << std::endl;
	delete[] this->vec;
	vec =  new ValType[othervec.capacity];
	if ( othervec.vec != nullptr && this->vec != othervec.vec && this->vec != nullptr )	
	{
		//delete[] this-> vec;
		for ( unsigned int i = 0; i < len_; i++ )
		{
			*( this->vec + i ) = *( othervec.vec + i );
		}
	}
}

//copy assignment operator
template < typename ValType >
Vector<ValType>& Vector< ValType >::operator=(const Vector& copysource )
{
	//std::cout << "copy assignment operator invoked " << std::endl;
	if ( this->vec != copysource.vec && copysource.vec != nullptr && this-> vec != nullptr )
	{
		delete[] this->vec;
		this->vec = new ValType [copysource.len_];
		this->len_ = copysource.len_;
		this->capacity = copysource.capacity;
		for (unsigned int i = 0; i < len_; i++)
		{
			*(vec + i) = *(copysource.vec + i);
		}
	}
	return *this;
}
//move constructor
template <typename ValType >
Vector< ValType >::Vector( Vector&& movesource ) noexcept
{
	//std::cout << "move constructor invoked" << std::endl;
	if ( movesource.vec != nullptr )
	{
		len_ = movesource.len_;
		capacity = movesource.capacity;
		vec = movesource.vec;
		movesource.vec = nullptr;
	}
}
//move asssignment operator
template < typename ValType >
Vector<ValType>& Vector< ValType >::operator=( Vector&& movesource ) noexcept
{
	if ( this != &movesource && movesource.vec != nullptr )
	{
		delete[] vec;
		this->vec  = movesource.vec;
		this->len_ = movesource.len_;
		this->capacity = movesource.capacity;
		movesource.vec = nullptr;
	}
	return *this;
}
//need anymore for the destructor?
template < typename ValType >
Vector<ValType>::~Vector()
{
	if ( vec != nullptr )
	{
		delete[] vec;
	}
	else 
	{
		vec = nullptr;
	}
}

template < typename ValType >
void Vector< ValType >::Show() const
{
	for (unsigned int i = 0; i < len_; i++)
	{
		std::cout << vec[i] << ", " << "";
	}
}

template < typename ValType >
Vector<ValType>::Vector( unsigned int user_size, ValType user_val)
{	
	realloc(user_size);
	for (unsigned int i = 0; i < user_size; i++)
	{
		vec[i] = user_val;
	}
	len_ = user_size;
}

//segmentation fault 
template < typename ValType >
Vector<ValType>::Vector(const ValType*  initialindex,  const ValType* finalindex)  
{
	size_t thesize = finalindex - initialindex + 1;
	//std::cout << "found the size of the vector" << std::endl;
	realloc( thesize );	
	//std::cout << "the capacity is: "<< capacity << std::endl;
	//std::cout << "reallocated to that size" << std::endl;
	len_ = thesize;
	//std::cout << "set the length to that size" << std::endl;
	int i = 0;
	for (const ValType* someptr = initialindex; someptr <= finalindex; someptr++)
	{
		*(vec + i) = *(initialindex+i);
		i ++;
	}
}

template < typename ValType >
Vector< ValType >::Vector( initializer_list< ValType > args )
{
	if ( len_ > 0 )
	{		
		for ( auto element:args )
		{
			this->push_back( element );
		}
	}
	else if ( len_ == 0 )
	{
		realloc(4);
		for( auto element : args )
		{
			this->push_back(element);
		}
	}
}

template < typename ValType >
inline ValType Vector<ValType>::getindex( unsigned int n ) const 
{	
	if ( n < len_ )
	{
		return *(vec+n);
	}
	else
	{
		throw(OutOfBoundsError( "Accessing element out of the bounds of your array." ));
	}
}

template < typename ValType >
inline void Vector<ValType>::setindex(const unsigned int& index, const ValType& valtobeput ) const
{	
	if ( index < len_ )
	{
		*( vec + index ) = valtobeput;
	}
	else 
	{
		throw(OutOfBoundsError(" The index you are trying to access must be consistent with the length of your array."));
	}
}

template < typename ValType >
double Vector<ValType>::Norm()
{
	double sum = 0;
	assert(len_ > 0);
	if (len_ == 1)
	{
		return *(vec);  
	}
	else if ( len_ == 2 )
	{
		return NormProxy2(double(vec[0]), double(vec[1]));		
	}
	else if ( len_ == 3 )
	{
		return NormProxy3(double(vec[0]), double(vec[1]), double(vec[2]));
	}
	else 
	{
		for ( unsigned int i = 0; i < len_; i ++)
		{	
			sum += std::pow( vec[i], 2);
		}
		return std::pow(sum, 0.5f);
	}

}

template < typename ValType >
double Vector< ValType >::lpnorm( double p )
{
	assert( p > 0 );
	double sum = 0;
	for ( unsigned int i = 0; i < len_; i++)
	{	
		sum += std::pow( vec[i], p );
		std::cout << sum << std::endl;
	}
	return std::pow( sum, (1.f/p) );
}

template < typename ValType >
ValType* Vector< ValType >::begin() const 
{	
	assert ( len_ > 0 && capacity > 0 );
	return vec;
}

template < typename ValType >
ValType* Vector< ValType >::end() const 
{	
	assert ( len_ > 0 && capacity > 0 );
	return (vec + (len_ - 1));
}

template < typename ValType > 
std::vector< ValType >& Vector<ValType>::GetTostdVec( std::vector< ValType >& temp ) const
{
	assert( len_ > 0 && capacity > 0 );
	assert( temp.size() == 0 );
	for ( unsigned int i = 0; i < len_; i++ )
	{
		temp.push_back(vec[i]);
	}
	return temp;
}

//addition operator not in class
template < typename ValType >
Vector< ValType > operator+(const Vector< ValType >& vec1, const Vector< ValType >& vec2 )
{
	if( vec1.size() != vec2.size() )
	{
		throw DimensionMismatch( " The two Vectors must have the same length. ");
	}
	else
	{
	Vector< ValType > temp = vec1;
	//assert( temp.size() == vec1.size() );
	ValType temp_val;
	for ( unsigned int i = 0 ; i < vec1.size(); i++)
	{
		temp_val = vec1.getindex(i)	+ vec2.getindex(i);
		temp.setindex( i, temp_val );
	}
	return temp;
	}
}

//Vector Subtraction
template < typename ValType >
Vector< ValType > operator-(const Vector< ValType >& vec1, const Vector< ValType >& vec2 )
{
	assert( vec1.size() == vec2.size() );
	Vector< ValType > temp = vec1;
	assert( temp.size() == vec1.size() );
	ValType temp_val;
	for ( unsigned int i = 0 ; i < vec1.size(); i++)
	{
		temp_val = vec1.getindex(i)	- vec2.getindex(i);
		temp.setindex( i, temp_val );
	}
	return temp;
}

//Dot product
template < typename ValType >
ValType operator *(const Vector< ValType >& vec1, const Vector< ValType >& vec2 )
{
	assert( vec1.size() == vec2.size() );
	ValType temp_val=0;
	for ( unsigned int i = 0 ; i < vec1.size(); i++)
	{
		temp_val += vec1.getindex(i) * vec2.getindex(i);
	}
	return temp_val;
}


template <typename T, typename Scalar>
Vector<T> operator*( const Vector<T>& vec, const Scalar& number )
{
	Vector<T> res(vec);
	for ( auto i = 0; i < vec.size(); ++i )
	{
		res.setindex( i, number*vec.getindex(i));
	}
	return res;
}
template <typename T, typename Scalar>
Vector<T> operator*( const Scalar& number, const Vector<T>& vec )
{
	return vec*number;
}

//need to refactor this to a cross(Vector<T> vec2) method 
//^^^
//done already.

template < typename ValType >
Vector< ValType > Vector< ValType >::cross( const Vector< ValType >& vec2 ) const
{
	assert(this->size()==vec2.size());
	assert(this->size()==2 || this->size()==3);
	ValType product,p1,p2,p3;
	if ( this->size()==2 )
	{   
		product = (this->getindex(0)*vec2.getindex(1))-(this->getindex(1)*vec2.getindex(0));
		return {0,0,product};
	}
	else if ( this->size() == 3 )
	{
        p1 = this->getindex(1)*vec2.getindex(2)-this->getindex(2)*vec2.getindex(1);
		p2 = this->getindex(0)*vec2.getindex(2)-this->getindex(2)*vec2.getindex(0);
		p3 = this->getindex(0)*vec2.getindex(1)-this->getindex(1)*vec2.getindex(0);
		return {p1,p2,p3};
	}
	else
	{
		std::cerr << "No cross products for vectors of size other than 2 and 3." << std::endl;
		return {-1,-1,-1};
	}
}






}


#endif
