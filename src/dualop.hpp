#ifndef DUALOP_H
#define DUALOP_H

//#include "dualad.hpp"
namespace linalg
{

template <typename T>
class NDual;


namespace dual_impl
{

//dual_impl::AssignmentOperation<T> is a pure interface, an abstract type 
//for all assignment operations
template<typename F, typename T>
struct ArithmeticOperation 
{
 	virtual ~ArithmeticOperation() = default;
	//pure virtual methods for the interface
 	virtual T& add_assign(const F& rhs_num) = 0;
 	virtual T& sub_assign(const F& rhs_num) = 0;
 	virtual T& mul_assign(const F& rhs_num) = 0;
 	virtual T& div_assign(const F& rhs_num) = 0;
 	virtual T& mod_assign(const F& rhs_num) = 0;
 	virtual T& add_op(const F& rhs_num) = 0;
 	virtual T& sub_op(const F& rhs_num) = 0;
 	virtual T& mul_op(const F& rhs_num) = 0;
 	virtual T& div_op(const F& rhs_num) = 0;
 	virtual T& mod_op(const F& rhs_num) = 0;
};

template < typename T >
struct NDualOperation// : public ArithmeticOperation<T,linalg::NDual<T>>
{
	//constructors
	NDualOperation()                                   = default; //default constructor
	NDualOperation( const NDualOperation& )            = default; //copy constructor
	NDualOperation( NDualOperation&& )                 = default; //move constructor
	NDualOperation& operator=( const NDualOperation& ) = default; //copy assignment operator
	NDualOperation& operator=( NDualOperation&& )      = default;      //move assignment operator
	virtual ~NDualOperation()                          = default;
	linalg::NDual<T>& add_assign( const T& rhs_num );    
	linalg::NDual<T>& sub_assign( const T& rhs_num );
	linalg::NDual<T>& mul_assign( const T& rhs_num );
	linalg::NDual<T>& div_assign( const T& rhs_num );
	linalg::NDual<T>& mod_assign( const T& rhs_num );
	
};










} //namespace dual_impl

} //namespace linalg
#endif
