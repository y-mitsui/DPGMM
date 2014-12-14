/* 2つのピークをもつ確率分布を混合ディリクレ過程で推定　*/
#include "dpgmm.h"
#include <stdlib.h>
#define DIMS 1
int main(void){
	gsl_rng *r = gsl_rng_alloc (gsl_rng_default);
	double weight[]={0.4,0.6},x;
	double mean[]={25.0,-1.0};
	double var[]={4.0,4.0};
	double num,stick;
	int i,j;
	char buf[128];
	
	FILE *fp=fopen("data.txt","w");
	for(i=0;i<2048;i++){
		stick=drand48();
		for(j=0;j<2;j++){
			stick-=weight[j];
			if(stick < 0.0) break;
		}
		num=var[j]*gsl_ran_gaussian(r,1)+mean[j];
		fprintf(fp,"%lf\n",num);
	}
	fclose(fp);
	fp=fopen("data.txt","r");
	DPGMM *ctx=dpgmm_init(1,8);
	while(fgets(buf,sizeof(buf),fp)){
		num=atof(buf);
		dpgmm_add(ctx,&num);
	}

	dpgmm_setDefaultsPrior(ctx);
	dpgmm_solv(ctx,1000);
	for(x=-10.0;x<40;x+=0.1){
		printf("%lf %lf\n",x,dpgmm_prob(ctx,&x));
	}
	return 0;
}
