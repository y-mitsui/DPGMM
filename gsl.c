#include "dpgmm.h"

void gsl_vector_print(gsl_vector *v){	
	int i;
	for(i=0;i<v->size;i++){
		printf("%.10lf ",gsl_vector_get(v,i));
	}
	puts("");
	
}
void gsl_matrix_print(gsl_matrix *m){	
	int i,j;
	for(i=0;i<m->size1;i++){
		for(j=0;j<m->size2;j++){
			printf("%.10lf ",gsl_matrix_get(m,i,j));
		}
		puts("");
	}
}
gsl_vector* gsl_matrix_sum_row(gsl_matrix *m){
	int i,j;
	gsl_vector* r=gsl_vector_alloc(m->size2);
	double total;
	for(i=0;i<m->size2;i++){
		total=0.0;
		for(j=0;j<m->size1;j++){
			total+=gsl_matrix_get(m,j,i);
		}
		gsl_vector_set(r,i,total);
	}
	return r;
}
double gsl_vector_sum(gsl_vector *v){
	int i;
	double r=0.0;
	for(i=0;i<v->size;i++){
		r+=gsl_vector_get(v,i);
	}
	return r;
}
gsl_vector *gsl_cumsum(gsl_vector *v){
	int i,j;
	gsl_vector *r=gsl_vector_alloc(v->size);
	gsl_vector_set_zero(r);
	for(i=0;i<v->size;i++){
		for(j=0;j<=i;j++){
			gsl_vector_set(r,i,gsl_vector_get(r,i)+gsl_vector_get(v,j));
		}
	}
	return r;
}
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
