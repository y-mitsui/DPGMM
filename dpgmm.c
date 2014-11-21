#include "dpgmm.h"

DPGMM *dpgmm_init(int dims,int stickCap){
	int i;
	DPGMM *r=calloc(1,sizeof(DPGMM));
	r->dims=dims;
	r->stickCap=stickCap;
	r->prior=gaussian_prior_init(dims);
	r->n=malloc(sizeof(GaussianPrior*)*stickCap);
	for(i=0;i<stickCap;i++)
		r->n[i]=gaussian_prior_init(dims);
	r->beta=gsl_vector_alloc(2);
	gsl_vector_set_all(r->beta,1.0);
	r->alpha=gsl_vector_alloc(2);
	gsl_vector_set_all(r->alpha,1.0);
	r->v=gsl_matrix_alloc(stickCap,2);
	gsl_matrix_set_all(r->v,1.0);
	r->z=NULL;
	r->skip=0;
	r->epsilon=1e-4;
	r->nT=calloc(1,sizeof(GaussianPrior*)*stickCap);
	r->vExpLog=gsl_vector_alloc(stickCap);
	r->vExpNegLog=gsl_vector_alloc(stickCap);
	gsl_vector_set_all(r->vExpNegLog,-1.0);
	gsl_vector_set_all(r->vExpLog,-1.0);
	r->data=malloc(sizeof(double)*LIMIT_DATA*dims);
	r->numData=0;
	return r;
}
void dpgmm_add(DPGMM *ctx,double *sample){
	memcpy(&ctx->data[ctx->numData*ctx->dims],sample,sizeof(double)*ctx->dims);
	ctx->numData++;
}
int setPrior(DPGMM *ctx,gsl_vector* mean,gsl_matrix* cover,double* weight,double scale){
	gaussian_prior_reset(ctx->prior);
	gaussian_prior_addPrior(ctx->prior,mean,cover,weight);
	ctx->priorT=gaussian_prior_intProb(ctx->prior);
	return 1;
}
int setDefaultsPrior(DPGMM *ctx){
	int i,j;
	gsl_vector *means=gsl_vector_alloc(ctx->dims);
	gsl_matrix *cover=gsl_matrix_alloc(ctx->dims,ctx->dims);
	for(i=0;i<ctx->dims;i++){
		gsl_vector_set(means,i,gsl_stats_mean(&ctx->data[i],ctx->dims,ctx->numData));
	}
	for(i=0;i<ctx->dims;i++){
		gsl_matrix_set(cover,i,i,gsl_stats_variance_m(&ctx->data[i],ctx->dims,ctx->numData,gsl_vector_get(means,i)));
		for(j=i+1;j<ctx->dims;j++){
			gsl_matrix_set(cover,i,j,gsl_stats_covariance_m(&ctx->data[i],ctx->dims,&ctx->data[j],ctx->dims,ctx->numData,gsl_vector_get(means,i),gsl_vector_get(means,j)));
			gsl_matrix_set(cover,j,i,gsl_matrix_get(cover,i,j));
		}
	}
	/*for(i=0;i<ctx->dims;i++){
		printf("means[%d]:%lf\n",i,gsl_vector_get(means,i));
	}
	for(i=0;i<ctx->dims;i++){
		for(j=0;j<ctx->dims;j++){
			printf("cover[%d,%d]:%lf\n",i,j,gsl_matrix_get(cover,i,j));
		}
	}*/
	setPrior(ctx,means,cover,NULL,1);
	return 1;
}
double *dpgmm_getDM(DPGMM *ctx){
	return ctx->data;
}
void dpgmm_solv(DPGMM *ctx){
	gsl_matrix *newZ;
	double *dm=dpgmm_getDM(ctx);
	
	if(ctx->z==NULL || ctx->z->size1<ctx->numData){
		newZ=gsl_matrix_alloc(ctx->numData,ctx->stickCap);
	}
}
void train(double *x,int numSample){
	int i;
	double beta[2]={1.0,1.0};
	double alpha[2];
	double vExpNegLog[1]={-1.0};
	double *newZ=malloc(sizeof(double)*DIMENTION);
	double *z,diff;
	double v[2],vExpLog;

	z=newZ;
	/*gsl_rng *r = gsl_rng_alloc (gsl_rng_default);	
	z=gsl_ran_dirichlet(r,1);*/
	for(i=0;i<DIMENTION;i++){
		z[i]=1.0;
	}
	//prev=z;
	do{
		alpha[0]=beta[0]+STICK_CAP;
		alpha[1]=beta[1]-vExpNegLog[0];
		double expLogStick=-gsl_sf_psi_int(1.0+alpha[0]/alpha[1]);
		double expNegLogStick = expLogStick;
		expLogStick += gsl_sf_psi_int(1.0);
		expNegLogStick += gsl_sf_psi_int(alpha[0]/alpha[1]);
		v[0]=1.0;
		v[1]=alpha[0]/alpha[1];
		double sums=sum(z,1);
		v[0]+=sums;
		v[1]+=DIMENTION;
		double *tmp=cumsum(&sums,1);
		v[1]=*tmp;
		vExpLog=-gsl_sf_psi_int(sum(v,2));
		vExpNegLog[0]= vExpLog;
		vExpLog+=gsl_sf_psi_int(v[0]);
		vExpNegLog[0]+= gsl_sf_psi_int(v[1]);
		for(i=0;i<STICK_CAP;i++){
			
		}
		diff=0.0;
	}while(diff<EPS);
}
/* TODO:double のNULLを定義*/
int main(void){
	double sample[2];
	char buf[128];

	DPGMM *ctx=dpgmm_init(DIMENTION,1);
	FILE *fp=fopen("data.txt","r");
	while(fgets(buf,sizeof(buf),fp)){
		sscanf(buf,"%lf,%lf",&sample[0],&sample[1]);
		dpgmm_add(ctx,sample);
	}
	fclose(fp);
	setDefaultsPrior(ctx);
	//train(sample,NUM_SAMPLE);
	return 0;
}


