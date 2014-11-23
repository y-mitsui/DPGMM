#include "dpgmm.h"


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

void student_t_setDOF(StudentT *ctx,double dof){
	ctx->dof=dof;
	ctx->norm=0.0;
}
double student_t_batchProb(StudentT *ctx,double *dm,int numData){
	int d=r->loc->size;
	double *delta;
	for(i=0;i<numData;i++){
		for(j=0;j<d;j++){
			delta[i*d+j]=dm[i*d+j]-ctx->loc[j];
		}
	}

	
}
void student_t_setLoc(StudentT *ctx,gsl_vector *loc){
	gsl_vector_memcpy(ctx->loc,loc);
}

void student_t_setInvScale(StudentT *ctx,gsl_matrix *invScale){
	ctx->scale=NULL;
	ctx->invScale=invScale;
	ctx->norm=0.0;
}
