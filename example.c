#include "dpgmm.h"
#include <time.h>
#define STICK_CAP 1
#define DIMENTION 2

int main(void){
	double sample[2];
	char buf[128];
	double train[2]={2.0,1.0};

	
	DPGMM *ctx=dpgmm_init(DIMENTION,STICK_CAP); //コンテキストを作成
	FILE *fp=fopen("data.txt","r");
	while(fgets(buf,sizeof(buf),fp)){
		sscanf(buf,"%lf,%lf",&sample[0],&sample[1]);
		dpgmm_add(ctx,sample);	//サンプリング
	}
	fclose(fp);
	
	clock_t t=clock();
	dpgmm_setDefaultsPrior(ctx); //事前分布の設定
	dpgmm_solv(ctx,10); //学習
	double p=dpgmm_prob(ctx,train); //事後確率
	printf("TIME:%.10f\n",(double)(clock()-t)/ (double)CLOCKS_PER_SEC);
	printf("RESULT:%lf\n",p);
	dpgmm_release(ctx);
	return 0;
}


