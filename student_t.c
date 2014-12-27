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
void student_t_free(StudentT *ctx){
	gsl_vector_free(ctx->loc);
	if(ctx->scale) gsl_matrix_free(ctx->scale);
	free(ctx);
}
double student_t_getLogNorm(StudentT *ctx){
	if(ctx->norm==0.0){
		int d=ctx->loc->size;
		ctx->norm=gsl_sf_lngamma(0.5*(ctx->dof+d));
		ctx->norm-=gsl_sf_lngamma(0.5*ctx->dof);
		ctx->norm -= log(ctx->dof*M_PI)*(0.5*d);
		int s;
		gsl_permutation *p = gsl_permutation_alloc (ctx->invScale->size1);
		gsl_matrix *lu=gsl_matrix_clone(ctx->invScale);
		gsl_linalg_LU_decomp (lu, p, &s);           // LU分解
		double n = gsl_linalg_LU_det (lu, s);    // 行列式
		ctx->norm += 0.5*log(n);

		gsl_matrix_free(lu);
		gsl_permutation_free(p);
	}
	return ctx->norm;
}
void student_t_setDOF(StudentT *ctx,double dof){
	ctx->dof=dof;
	ctx->norm=0.0;
}
double student_t_prob(StudentT *ctx,const double *x){
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

	gsl_matrix_free(delta);
	gsl_matrix_free(tmp);
	gsl_matrix_free(tmp2);
	double tmp3=log(val)*(-0.5*(ctx->dof+d));
	return exp(student_t_getLogNorm(ctx) + tmp3);
}
void student_t_batchProb(StudentT *ctx,double *dm,int numData,double *result){
	int d=ctx->loc->size,i,j,k;
	double *delta=malloc(sizeof(double)*numData*d);
	
	
	for(i=0;i<numData;i++){
		for(j=0;j<d;j++){
			delta[i*d+j]=dm[i*d+j]-gsl_vector_get(ctx->loc,j);
		}
	}
	/*puts("ctx->invScale:");
	gsl_matrix_print(ctx->invScale);*/
	double *tmp=calloc(1,sizeof(double)*numData*d);
	for(i=0;i<numData;i++){
		for(j=0;j<d;j++){
			for(k=0;k<d;k++){
				tmp[i*d+j]+=gsl_matrix_get(ctx->invScale,j,k)*delta[i*d+k];
			}
		}
	}
	for(i=0;i<numData;i++){
		for(j=0;j<d;j++){
			result[i]+=tmp[i*d+j]*delta[i*d+j];
		}
		//printf("result[%d]:%lf\n",i,result[i]);
	}
	//puts("");
	for(i=0;i<numData;i++){
		result[i]=1.0+result[i]/ctx->dof;
	}
	for(i=0;i<numData;i++){
		//printf("result[i]:%lf ",result[i]);
		result[i]=exp(student_t_getLogNorm(ctx)+log(result[i])*(-0.5*(ctx->dof+d)));
		//printf("result[i]:%lf\n",result[i]);
	}

	free(delta);
	free(tmp);
}
void student_t_setLoc(StudentT *ctx,gsl_vector *loc){
	gsl_vector_memcpy(ctx->loc,loc);
}

void student_t_setInvScale(StudentT *ctx,gsl_matrix *invScale){
	if(ctx->scale) gsl_matrix_free(ctx->scale);
	ctx->scale=NULL;
	ctx->invScale=invScale;
	ctx->norm=0.0;
}
