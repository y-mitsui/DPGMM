CFLAGS=-Wall `gsl-config --cflags` -g
LOADLIBES=`gsl-config --libs`
dpgmm: gaussian_prior.o gsl.o calc.o student_t.o
