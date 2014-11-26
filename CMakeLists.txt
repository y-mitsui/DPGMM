cmake_minimum_required (VERSION 2.6)

project (DpBayesianNet)

SET(CMAKE_C_FLAGS "-DHAVE_INLINE -O2 -Wall -g -I/usr/include -I/usr/local/include")

ADD_LIBRARY(dpgmm STATIC dpgmm.c gaussian_prior.c gsl.c calc.c student_t.c)


INSTALL(TARGETS dpgmm ARCHIVE DESTINATION lib)
INSTALL(FILES dpgmm.h DESTINATION include)

add_executable (example example.c)
target_link_libraries(example dpgmm m -L/usr/local/lib -lgsl  -lgslcblas)
