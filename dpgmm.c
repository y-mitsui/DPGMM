#include "dpgmm.h"
#include <time.h>

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
	r->nT=calloc(1,sizeof(StudentT*)*stickCap);
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
	double *alpha,*theta,expLogStick,expNegLogStick;
	int offset,i,j;
	
	if(ctx->z==NULL || ctx->z->size1<ctx->numData){
		newZ=gsl_matrix_alloc(ctx->numData,ctx->stickCap);
		if(ctx->z==NULL) offset=0;
		else{
			offset=ctx->z->size1;
			gsl_vector *v=gsl_vector_alloc(ctx->z->size2);
			for(i=0;i<offset;i++){	/*TODO:gsl matrix view*/
				gsl_matrix_get_row(v,ctx->z,i);
				gsl_matrix_set_row(newZ,i,v);
			}
		}
		ctx->z=newZ;
		
		alpha=malloc(sizeof(double)*ctx->stickCap);
		theta=malloc(sizeof(double)*ctx->stickCap);
		for(i=0;i<ctx->stickCap;i++){
			alpha[i]=32.0;
		}

		if(ctx->stickCap==1){
			gsl_matrix_set_all(ctx->z,1.0);
		}else{
			size_t size=ctx->z->size1-offset;
			for(i=0;i<size;i++){
				gsl_rng *r = gsl_rng_alloc (gsl_rng_default);
				gsl_ran_dirichlet(r,ctx->stickCap,alpha,theta);
				for(j=0;j<ctx->stickCap;j++){
					gsl_matrix_set(ctx->z,i,j,theta[j]);
				}
			}
		}

		gsl_matrix *prev=gsl_matrix_clone(ctx->z);
		int iters=0;
		do{
			gsl_vector_set(ctx->alpha,0,gsl_vector_get(ctx->beta,0)+ctx->stickCap);
			gsl_vector_set(ctx->alpha,1,gsl_vector_get(ctx->beta,1)-gsl_vector_sum(ctx->vExpNegLog));
			double alphaRate=gsl_vector_get(ctx->alpha,0)/gsl_vector_get(ctx->alpha,1);
			expLogStick=-gsl_sf_psi(1.0 + alphaRate);
			expNegLogStick=expLogStick;
			expLogStick += gsl_sf_psi(1.0);
			expNegLogStick +=gsl_sf_psi(alphaRate);

			for(i=0;i<ctx->v->size1;i++){
				gsl_matrix_set(ctx->v,i,0,1.0);
				gsl_matrix_set(ctx->v,i,1,alphaRate);
			}

			gsl_vector *sums = gsl_matrix_sum_row(ctx->z);
			gsl_vector *cumsums = gsl_cumsum(sums);
			for(i=0;i<ctx->v->size1;i++){
				gsl_matrix_set(ctx->v,i,0,gsl_matrix_get(ctx->v,i,0)+gsl_vector_get(sums,i));
				gsl_matrix_set(ctx->v,i,1,gsl_matrix_get(ctx->v,i,1)+ctx->z->size1);
				gsl_matrix_set(ctx->v,i,1,gsl_matrix_get(ctx->v,i,1)-gsl_vector_get(cumsums,i));
			}

			for(i=0;i<ctx->v->size1;i++){
				double total=0.0;
				for(j=0;j<ctx->v->size1;j++){
					total+=gsl_matrix_get(ctx->z,i,j);
				}
				gsl_vector_set(ctx->vExpLog,i,-gsl_sf_psi(total));
			}
			gsl_vector_memcpy(ctx->vExpNegLog,ctx->vExpLog);
			for(i=0;i<ctx->v->size1;i++){
				gsl_vector_set(ctx->vExpLog,i,gsl_vector_get(ctx->vExpLog,i)+gsl_sf_psi(gsl_matrix_get(ctx->v,i,0)));
				gsl_vector_set(ctx->vExpNegLog,i,gsl_vector_get(ctx->vExpNegLog,i)+gsl_sf_psi(gsl_matrix_get(ctx->v,i,1)));
			}
			for(i=0;i<ctx->stickCap;i++){
				gaussian_prior_reset(ctx->n[i]);
				gaussian_prior_addGP(ctx->n[i],ctx->prior);
				double *weight=malloc(sizeof(double)*ctx->z->size1);
				for(j=0;j<ctx->z->size1;j++){
					weight[j]=gsl_matrix_get(ctx->z,j,i);
				}
				gaussian_prior_addSamples(ctx->n[i],dm,ctx->numData,weight);
				ctx->nT[i]=gaussian_prior_intProb(ctx->n[i]);
			}

			gsl_vector *v=gsl_vector_alloc(ctx->z->size2);/*TODO:gsl matrix view*/
			for(i=ctx->skip;i<prev->size1;i++){
				gsl_matrix_get_row(v,ctx->z,i);
				gsl_matrix_set_row(prev,i,v);
			}
			
			gsl_vector *vExpNegLogCum=gsl_cumsum(ctx->vExpNegLog);
			gsl_vector *base=gsl_vector_clone(ctx->vExpLog);
			
			for(i=1;i<vExpNegLogCum->size;i++){
				gsl_vector_set(base,i,gsl_vector_get(vExpNegLogCum,i-1));
			}
			gsl_vector *expTmp=gsl_vector_alloc(base->size);
			for(i=0;i<base->size;i++){
				gsl_vector_set(expTmp,i,exp(gsl_vector_get(base,i)));
			}
			for(i=ctx->skip;i<ctx->z->size1;i++){
				gsl_matrix_set_row(ctx->z,i,expTmp);
			}
			for(i=0;i<ctx->stickCap;i++){
				double *val=student_t_batchProb(ctx->nT[i],&dm[ctx->skip*ctx->dims],ctx->numData-ctx->skip);
				for(j=0;j<ctx->numData-ctx->skip;j++){
					gsl_matrix_set(ctx->z,ctx->skip+j,i,gsl_matrix_get(ctx->z,ctx->skip+j,i)+val[i]);
				}
			}
			double *norm=student_t_batchProb(ctx->priorT,&dm[ctx->skip*ctx->dims],ctx->numData-ctx->skip);
			for(i=0;i<ctx->numData-ctx->skip;i++){
				norm[i]*=exp(expLogStick + gsl_vector_get(vExpNegLogCum,vExpNegLogCum->size-1)) / (1.0 - exp(expNegLogStick));
			}
		}while(++iters<4);
		
	}
	
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
	
	clock_t t=clock();
	setDefaultsPrior(ctx);	
	dpgmm_solv(ctx);
	printf("%.10f\n",(double)(clock()-t)/ (double)CLOCKS_PER_SEC);
	//train(sample,NUM_SAMPLE);
	return 0;
}


