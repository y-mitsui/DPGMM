#include "dpgmm.h"
#include <stdlib.h>
#define DIMS 2
int main(void){
	gsl_rng *r = gsl_rng_alloc (gsl_rng_default);
	double x,y;
	double sample[2];
	double num;
	int i;
	char buf[128];
	
	FILE *fp=fopen("data4.txt","r");
	DPGMM *ctx=dpgmm_init(DIMS,8);
	while(fgets(buf,sizeof(buf),fp)){
		sscanf(buf,"%lf %lf",&sample[0],&sample[1]);
		dpgmm_add(ctx,sample);
	}
	dpgmm_setDefaultsPrior(ctx);
	dpgmm_solv(ctx,1000);
	for(x=-1.0;x<18;x+=0.1){
		sample[0]=x;
		for(y=0.95;y<1.05;y+=0.001){
			sample[1]=y;
			printf("%lf %lf %lf\n",x,y,dpgmm_prob(ctx,sample));
		}
	}
	return 0;
}
