#ifndef EXP_H
#define EXP_H

#include <iostream>
namespace linalg
{

class OutOfBoundsError : public std::exception
{

private:
	std::string m_error{};
public:
	OutOfBoundsError( std::string error ) 
		:m_error{ error }
	{}

	const char* what() const noexcept override { return m_error.c_str(); }
};

class DimensionMismatch : public std::exception
{

private:
	std::string m_error{};
public:
	DimensionMismatch( std::string error ) 
		:m_error{ error }
	{}

	const char* what() const noexcept override { return m_error.c_str(); }
};

class NullSizeError : public std::exception
{

private: 
	std::string m_error{};
public:
	NullSizeError( std::string error )
		:m_error{ error }
	{}

	const char* what() const noexcept override { return m_error.c_str(); }
};

class MemoryAssignmentError : public std::exception
{

private:
	std::string m_error{};
public:
	MemoryAssignmentError( std::string error ) 
		:m_error{ error }
	{}

	const char* what() const noexcept override { return m_error.c_str(); }
};

class DualError : public std::exception
{

private:
	std::string m_error{};
public:
	DualError( std::string error ) 
		:m_error{ error }
	{}

	const char* what() const noexcept override { return m_error.c_str(); }
};

class MatrixError : public std::exception
{

private:
	std::string m_error{};
public:
	MatrixError( std::string error ) 
		:m_error{ error }
	{}

	const char* what() const noexcept override { return m_error.c_str(); }
};


}

#endif
