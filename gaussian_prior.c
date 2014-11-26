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
void gaussian_prior_free(GaussianPrior *ctx){
	gsl_matrix_free(ctx->invShape);
	gsl_vector_free(ctx->mu);
	if(ctx->shape) gsl_matrix_free(ctx->shape);
	free(ctx);
}
gsl_matrix *gaussian_prior_getLambda(GaussianPrior *ctx){
	if(ctx->shape==NULL){
		gsl_matrix *invShape=gsl_matrix_clone(ctx->invShape);
		gsl_permutation * p = gsl_permutation_alloc (invShape->size1);
		int s;
		gsl_linalg_LU_decomp(invShape,p,&s);
		ctx->shape=gsl_matrix_alloc(ctx->invShape->size1,ctx->invShape->size2);
		gsl_linalg_LU_invert(invShape,p,ctx->shape);
		gsl_matrix_free(invShape);
		gsl_permutation_free(p);
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
void gaussian_prior_addSamples(GaussianPrior *ctx,double *sample,int numSample,double *weight){
	int d=ctx->mu->size,i,j,k;
	double num=sum(weight,numSample);
	double *means=malloc(sizeof(double)*d);
	double *diff=malloc(sizeof(double)*d);
	double *scatter=malloc(sizeof(double)*d*d);
	for(i=0;i<d;i++){
		means[i]=gsl_stats_mean(&sample[i],d,numSample);
	}
	for(i=0;i<numSample;i++){
		for(j=0;j<d;j++){
			diff[j]=sample[i*d+j]-means[j];
		}
		for(j=0;j<d;j++){
			for(k=0;k<d;k++){
				scatter[j*d+k]+=weight[i]*diff[j]*diff[k];
			}
		}
	}
	
	gsl_vector *delta=gsl_vector_alloc(d);
	for(i=0;i<d;i++){
		gsl_vector_set(delta,i,means[i]-gsl_vector_get(ctx->mu,i));
	}
	gsl_matrix *tmp=gsl_vector_outer(delta,delta);
	for(i=0;i<d;i++){
		for(j=0;j<d;j++){
			double n=scatter[i*d+j]+gsl_matrix_get(tmp,i,j)*(ctx->k*num)/(ctx->k+num);
			gsl_matrix_set(ctx->invShape,i,j,gsl_matrix_get(ctx->invShape,i,j)+n);
		}
	}
	ctx->shape=NULL;
	gsl_vector_mul_constant(delta,num/(ctx->k+num));
	gsl_vector_add(ctx->mu,delta);
	ctx->n+=num;
	ctx->k+=num;
	free(means);
	free(diff);
	free(scatter);
}

void gaussian_prior_addGP(GaussianPrior *ctx,GaussianPrior *gp){
	gsl_vector *delta=gsl_vector_clone(gp->mu);
	gsl_vector_sub(delta,ctx->mu);
	double tmp=((gp->k*ctx->k)/(gp->k+ctx->k));
	int i,j;
	for(i=0;i<ctx->invShape->size1;i++){
		for(j=0;j<ctx->invShape->size1;j++){
			gsl_matrix_set(ctx->invShape,i,j,gsl_matrix_get(ctx->invShape,i,j)+gsl_matrix_get(gp->invShape,i,j)+tmp);
		}
	}
	/*ctx->invShape+=gp->invShape;
	ctx->invShape+=((gp->k*ctx->k)/(gp->k+ctx->k));*/
	ctx->shape=NULL;
	tmp=gp->k/(ctx->k+gp->k);
	for(i=0;i<ctx->mu->size;i++){
		gsl_vector_set(ctx->mu,i,gsl_vector_get(ctx->mu,i)+tmp+gsl_vector_get(delta,i));
	}
	ctx->n += gp->n;
	ctx->k += gp->k;
	gsl_vector_free(delta);
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
	gsl_vector_free(delta);
}
