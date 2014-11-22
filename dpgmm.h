#ifndef _DPGMM_H
#define _DPGMM_H 1

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_linalg.h>

#include <stdio.h>
#include <string.h>

#define STICK_CAP 1
#define LIMIT_DATA 40000
#define DIMENTION 2
#define EPS 0.1
#define NUM_SAMPLE 10
typedef struct {
	gsl_matrix *invShape;
	gsl_matrix *shape; //cache
	gsl_vector *mu;
	double n,k;
} GaussianPrior;
typedef struct{
	double dof;
	gsl_vector *loc;
	gsl_matrix *scale;
	gsl_matrix *invScale;
	double norm;	
}StudentT;
typedef struct{
	int dims;
	int stickCap;
	GaussianPrior *prior;
	StudentT *priorT;
	GaussianPrior **n;
	gsl_vector *beta;
	gsl_vector *alpha;
	gsl_matrix *v;
	gsl_matrix *z;
	int skip;
	double epsilon;
	GaussianPrior **nT;
	gsl_vector *vExpLog;
	gsl_vector *vExpNegLog;
	double *data;
	int numData;
} DPGMM;


void setArray(double *array,int num,double val);
double sum(double *array,int num);
double *cumsum(double *array,int num);

gsl_vector *gsl_vector_clone(gsl_vector *src);
gsl_matrix *gsl_matrix_clone(gsl_matrix *src);
void gsl_matrix_mul_constant(gsl_matrix *a,const double x);
void gsl_vector_mul_constant(gsl_vector *a,const double x);
gsl_matrix* gsl_vector_outer(gsl_vector *a,gsl_vector *b);
double gsl_vector_sum(gsl_vector *v);
gsl_vector *gsl_cumsum(gsl_vector *v);
gsl_vector* gsl_matrix_sum_row(gsl_matrix *m);


GaussianPrior *gaussian_prior_init(int dims);
void gaussian_prior_reset(GaussianPrior *ctx);
void gaussian_prior_addPrior(GaussianPrior *ctx,gsl_vector *mean,gsl_matrix *covariance,double* weight);
StudentT * gaussian_prior_intProb(GaussianPrior *ctx);
void gaussian_prior_addGP(GaussianPrior *ctx,GaussianPrior *gp);

StudentT *student_t_init(int dims);
void student_t_setDOF(StudentT *ctx,double dof);
void student_t_setLoc(StudentT *ctx,gsl_vector *loc);
void student_t_setInvScale(StudentT *ctx,gsl_matrix *invScale);

#endif
