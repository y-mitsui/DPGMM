#include "dpgmm.h"
#include <gsl/gsl_blas.h>

StudentT *student_t_init(int dims){
	StudentT *r=malloc(sizeof(StudentT));
	r->dof=1.0;
	r->loc=gsl_vector_calloc(dims);
	r->scale=gsl_matrix_alloc(dims,dims);
	gsl_matrix_set_identity(r->scale);
	r->invScale=NULL;
	r->norm=0.0;
	return r;
}
double student_t_getLogNorm(StudentT *ctx){
	if(ctx->norm==0.0){
		int d=ctx->loc->size;
		ctx->norm=gsl_sf_lngamma(0.5*(ctx->dof+d));
		ctx->norm-=gsl_sf_lngamma(0.5*ctx->dof);
		ctx->norm -= log(ctx->dof*M_PI)*(0.5*d);
		int             s;
		gsl_permutation *p = gsl_permutation_alloc (ctx->invScale->size1);
		gsl_matrix *lu=gsl_matrix_clone(ctx->invScale);
		gsl_linalg_LU_decomp (lu, p, &s);           // LU分解
		double n = gsl_linalg_LU_det (lu, s);    // 行列式
		ctx->norm += 0.5*log(n);
	}
	return ctx->norm;
}
void student_t_setDOF(StudentT *ctx,double dof){
	ctx->dof=dof;
	ctx->norm=0.0;
}
double student_t_prob(StudentT *ctx,double *x){
	int d = ctx->loc->size,i;
	gsl_matrix *delta=gsl_matrix_alloc(d,1);
	gsl_matrix *tmp=gsl_matrix_alloc(d,1);
	gsl_matrix *tmp2=gsl_matrix_alloc(1,1);
	for(i=0;i<d;i++){
		gsl_matrix_set(delta,i,0,x[i]-gsl_vector_get(ctx->loc,i));
	}
	gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                  1.0, ctx->invScale, delta,
                  0.0, tmp);
	gsl_blas_dgemm (CblasTrans, CblasNoTrans,
                  1.0, delta, tmp,
                  0.0, tmp2);
	double val=gsl_matrix_get(tmp2,0,0);
	val = 1.0 + val/ctx->dof;
    	return exp(student_t_getLogNorm(ctx) + log(val)*(-0.5*(ctx->dof+d)));
}
double *student_t_batchProb(StudentT *ctx,double *dm,int numData){
	int d=ctx->loc->size,i,j,k;
	double *delta=malloc(sizeof(double)*numData*d);
	for(i=0;i<numData;i++){
		for(j=0;j<d;j++){
			delta[i*d+j]=dm[i*d+j]-gsl_vector_get(ctx->loc,j);
		}
	}
	double *tmp=calloc(1,sizeof(double)*numData*d*d);
	for(i=0;i<numData;i++){
		for(j=0;j<d;j++){
			for(k=0;k<d;k++){
				tmp[i*d+j]+=gsl_matrix_get(ctx->invScale,j,k)*delta[i*d+k];
			}
		}
	}
	double *val=calloc(1,sizeof(double)*numData);
	for(i=0;i<numData;i++){
		for(j=0;j<d;j++){
			val[i]+=tmp[i*d+j]*delta[i*d+j];
		}
	}
	for(i=0;i<numData;i++){
		val[i]=1.0+val[i]/ctx->dof;
	}
	for(i=0;i<numData;i++){
		val[i]=exp(student_t_getLogNorm(ctx)+log(val[i])*(-0.5*(ctx->dof+d)));
	}
	return val;
}
void student_t_setLoc(StudentT *ctx,gsl_vector *loc){
	gsl_vector_memcpy(ctx->loc,loc);
}

void student_t_setInvScale(StudentT *ctx,gsl_matrix *invScale){
	ctx->scale=NULL;
	ctx->invScale=invScale;
	ctx->norm=0.0;
}
