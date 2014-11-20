#include "dpgmm.h"

gsl_vector *gsl_vector_clone(gsl_vector *src){
	gsl_vector *r=gsl_vector_alloc(src->size);
	gsl_vector_memcpy(r,src);
	return r;
}
gsl_matrix *gsl_matrix_clone(gsl_matrix *src){
	gsl_matrix *r=gsl_matrix_alloc(src->size1,src->size2);
	gsl_matrix_memcpy(r,src);
	return r;
}

void gsl_matrix_mul_constant(gsl_matrix *a,const double x){
	int i,j;
	for(i=0;i<a->size1;i++){
		for(j=0;j<a->size2;j++){
			gsl_matrix_set(a,i,j,gsl_matrix_get(a,i,j)*x);
		}
	}
}
void gsl_vector_mul_constant(gsl_vector *a,const double x){
	int i;
	for(i=0;i<a->size;i++){
		gsl_vector_set(a,i,gsl_vector_get(a,i)*x);
	}
}
gsl_matrix* gsl_vector_outer(gsl_vector *a,gsl_vector *b){
	int i,j;
	gsl_matrix *r=gsl_matrix_alloc(a->size,b->size);
	for(i=0;i<r->size1;i++){
		for(j=0;j<r->size2;j++){
			gsl_matrix_set(r,i,j,gsl_vector_get(a,i)*gsl_vector_get(b,j));
		}
	}
	return r;
}
