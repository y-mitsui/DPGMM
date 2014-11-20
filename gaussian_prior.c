#include "dpgmm.h"

GaussianPrior *gaussian_prior_init(int dims){
	GaussianPrior *r=calloc(1,sizeof(GaussianPrior));
	r->invShape=gsl_matrix_alloc(dims,dims);
	gsl_matrix_set_zero(r->invShape);
	r->shape=NULL;
	r->mu=gsl_vector_alloc(dims);
	gsl_vector_set_zero(r->mu);
	r->n=r->k=0.0;
	return r;
}
gsl_matrix *gaussian_prior_getLambda(GaussianPrior *ctx){
	if(ctx->shape==NULL){
		gsl_matrix *invShape=gsl_matrix_clone(ctx->invShape);
		gsl_permutation * p = gsl_permutation_alloc (invShape->size1);
		int s;
		gsl_linalg_LU_decomp(invShape,p,&s);
		ctx->shape=gsl_matrix_alloc(ctx->invShape->size1,ctx->invShape->size2);
		gsl_linalg_LU_invert(invShape,p,ctx->shape);
	}
	return ctx->shape;
}
void gaussian_prior_reset(GaussianPrior *ctx){
	gsl_matrix_set_zero(ctx->invShape);
	ctx->shape=NULL;
	gsl_vector_set_zero(ctx->mu);
	ctx->n=ctx->k=0.0;
}

StudentT * gaussian_prior_intProb(GaussianPrior *ctx){
	int d=ctx->mu->size;
	StudentT *st=student_t_init(d);
	double dof=ctx->n-d+1.0;
	student_t_setDOF(st,dof);
	student_t_setLoc(st,ctx->mu);
	double mult=ctx->k*dof/ (ctx->k+1.0);
	gsl_matrix  *lambda=gaussian_prior_getLambda(ctx);
	gsl_matrix_mul_constant(lambda,mult);
	student_t_setInvScale(st,lambda);
	return st;
}

void gaussian_prior_addPrior(GaussianPrior *ctx,gsl_vector *mean,gsl_matrix *covariance,double* weight){
	double weight_const;
	weight_const=(weight==NULL)  ? (double)ctx->mu->size : *weight;
	gsl_vector *delta=gsl_vector_clone(mean);
	gsl_vector_sub(delta,ctx->mu);

	gsl_matrix *tmp=gsl_matrix_clone(covariance);
	gsl_matrix_mul_constant(tmp,weight_const);
	gsl_matrix_add(ctx->invShape,tmp);

	tmp=gsl_vector_outer(delta,delta);
	gsl_matrix_mul_constant(tmp,(ctx->k*weight_const)/(ctx->k+weight_const));
	ctx->shape=NULL;
	gsl_vector_mul_constant(delta,weight_const/(ctx->k+weight_const));
	gsl_vector_add(ctx->mu,delta);
	ctx->n+=weight_const;
	ctx->k+=weight_const;
}
