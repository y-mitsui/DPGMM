#include "dpgmm.h"

void setArray(double *array,int num,double val){
	int i;
	for(i=0;i<num;i++){
		array[i]=val;
	}
}
double sum(double *array,int num){
	double r=0.0;
	int i;
	for(i=0;i<num;i++){
		r+=array[i];
	}
	return r;
}
double *cumsum(double *array,int num){
	int i,j;
	double *r=calloc(1,sizeof(double)*num);

	for(i=0;i<num;i++){
		for(j=0;j<i;j++){
			r[i]+=array[i];
		}
	}
	return r;
}
