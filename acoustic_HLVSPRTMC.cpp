#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define PI 3.1415926

#include "mpi.h"
#include "headrtm.h"
#include "fftw3.h"

#define ricker_flag 1

int main(int argc, char *argv[])
{
	int myid,numprocs,namelen,index;
	
	MPI_Comm comm=MPI_COMM_WORLD;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(comm,&myid);
	MPI_Comm_size(comm,&numprocs);
	MPI_Get_processor_name(processor_name,&namelen);

	if(myid==0)
		printf("Number of MPI thread is %d\n",numprocs);

	/*=========================================================
	  Parameters of the time of the system...
	  =========================================================*/
	time_t begin_time;
	//  time_t end_time;
	//  time_t last_time;

	clock_t start;
	clock_t end;

	//  float runtime=0.0;

	int nx,nz;
	int pml,Lc;

	float dx,dz;

	float rectime,dt;
	int   itmax;
	float f0;

	int ns,r_n;
	float sx0,shotdx,shotdep,rx0,recdx,recdix,offsetz;

	int loop;
	int Ns; 
	int i,ii,ic,GPU_N;
	int scatter_flag,swin;
	int window;

	char v_file[150],data_file[150],srn_file[150];
	char sx_file[150],srx_file[150],mx_file[150];
	char scatter_file[150];

	input_parameters(&nx,&nz,&pml,&Lc,&dx,&dz,&itmax,&dt,&f0,&ns,&sx0,
			&shotdx,&shotdep,&recdix,&offsetz,&scatter_flag,&swin,&window);//;

	/*=========================================================
	  File name....
	 *========================================================*/

	FILE *fp;
	char filename[50];

	/*=========================================================
	  Parameters of Cartesian coordinate...
	  ========================================================*/  

	int pmlc=pml+Lc;

	int ntz=nz+2*pmlc;
	int ntx=nx+2*pmlc;
	int ntp=ntz*ntx;
	int np=nx*nz;

	int ip,ipp,iz,ix,it;

	int *mx=(int*)malloc(sizeof(int)*ntx);
	for(ix=0;ix<ntx;ix++)
		mx[ix]=ix;

	/*=========================================================
	  Parameters of ricker wave...
	  ========================================================*/

	float *rick,*hrick;
	float t0=1.0/f0;
	int t0n=0;//(t0-1.0/20.0)/dt;

	if(myid==0)
		printf("The simulation time step is %d\n",itmax);

	/*=========================================================
	  Parameters of ricker wave...
	  ========================================================*/

	float *rc;
	rc=(float*)malloc(sizeof(float)*Lc);
	cal_xishu(Lc,rc);

	float tmprc=0.0;
	for(ic=0;ic<Lc;ic++)
	{
		tmprc+=fabs(rc[ic]);
	}
	if(myid==0)
	{
		printf("Maximum velocity for stability is %f m/s\n",dx/(tmprc*dt*sqrt(2)));
	}

	/*=========================================================
	  Parameters of GPU...
	  ========================================================*/

	getdevice(&GPU_N);
	int GPU_NN=GPU_N;
	printf("The available Device number is %d on %s\n",GPU_N,processor_name);
	MPI_Barrier(comm);

	struct MultiGPU plan[GPU_N];

	if(ns%GPU_N!=0)
	{
		printf("Source number cannot be divided by GPUs !\n");
		printf("Break up");
		return(0);
	}

	/*=========================================================
	  Parameters of Sources and Receivers...
	  ========================================================*/
	int is,rnmax=0;

	int nsid,modsr,prcs,prcss;
	int iss,eachsid,offsets;

	/*=========================================================
	  Calculate the sourcess' poisition...
	  ========================================================*/

	struct Source ss[ns];

	///////////////////////////////////////////////////////
	for(is=0;is<ns;is++)
	{
		ss[is].s_ix=pmlc+(int)(sx0/dx)+(int)(shotdx/dx)*is;//29+is*55;//

		ss[is].s_iz=pmlc+(int)(shotdep/dz);
		ss[is].r_ix=pmlc+nx/2;//(int)(recdix/dx);

		//ss[is].r_n=srn[is];//r_n;
		i=0;
		for(iz=0;iz<nz;iz++)
		{
			if(fabs(ss[is].s_iz-iz-pmlc)*dz<=offsetz)
				i++;
		}
		ss[is].r_n=2*i;
	}

	for(is=0;is<ns;is++)
	{
		if(rnmax<ss[is].r_n)
			rnmax=ss[is].r_n;
	}
	if(myid==0)
		printf("The maximum trace number for source is %d\n",rnmax);

	for(is=0;is<ns;is++)
	{
		ss[is].r_iz=(int*)malloc(sizeof(int)*ss[is].r_n);
	} 

	for(is=0;is<ns;is++)
	{
		i=0;
		for(iz=0;iz<nz;iz++)
		{
			if(fabs(ss[is].s_iz-iz-pmlc)*dz<=offsetz)
			{
				ss[is].r_iz[i]=pmlc+iz;
				ss[is].r_iz[i+ss[is].r_n/2]=pmlc+iz;
				i++;
			}
		}
		if(i>ss[is].r_n)
		{
			printf("The trace number of %d th source is out of range!\n",is+1);
			return(0);
		}
	}

	/*=========================================================
	  Parameters of model...
	  ========================================================*/

	float *vp,*rho;
	float *vpn,*rhon;
	float vp_max,rho_max;
	float vp_min,rho_min;

	/*=========================================================
	  Parameters of absorbing layers...
	  ========================================================*/

	//  float thickness_of_pmlc=pmlc*dx;
	//  float Rc=1.0e-3;
	//  float Vpmax=3300.0;
	//  float d0;

	float *d_x,*d_x_half,*d_z,*d_z_half;
	float *a_x,*a_x_half,*a_z,*a_z_half;
	float *b_x,*b_x_half,*b_z,*b_z_half;
	float *k_x,*k_x_half,*k_z,*k_z_half;

	/*=========================================================
	  Parameters of the coefficients of the space...
	  ========================================================*/

	float c[2]={9.0/8.0,-1.0/24.0};

	/*=========================================================
	  Image / gradient ...
	 *========================================================*/

	int thetan=10,delt=1;

	float *migration_pp;
	float *migration_ppt;

	float *migration_ppcig;

	float *tmp1,*tmp2,*tmp3;

	/*=========================================================
	  Flags ....
	 *========================================================*/

	int inv_flag;

	//#######################################################################
	// NOW THE PROGRAM BEGIN
	//#######################################################################

	time(&begin_time);
	if(myid==0)
		printf("Today's data and time: %s",ctime(&begin_time));

	/*=========================================================
	  Allocate the memory of parameters of ricker wave...
	  ========================================================*/

	rick=(float*)malloc(sizeof(float)*itmax);
	hrick=(float*)malloc(sizeof(float)*itmax);

	/*=========================================================
	  Allocate the memory of parameters of model...
	  ========================================================*/

	// allocate the memory of model parameters...

	vp                  = (float*)malloc(sizeof(float)*ntp);
	rho                 = (float*)malloc(sizeof(float)*ntp);

	vpn                  = (float*)malloc(sizeof(float)*ntp);
	rhon                 = (float*)malloc(sizeof(float)*ntp);

	/*=========================================================
	  Allocate the memory of parameters of absorbing layer...
	  ========================================================*/

	d_x      = (float*)malloc(ntx*sizeof(float));
	d_x_half = (float*)malloc(ntx*sizeof(float));    
	d_z      = (float*)malloc(ntz*sizeof(float));
	d_z_half = (float*)malloc(ntz*sizeof(float));


	a_x      = (float*)malloc(ntx*sizeof(float));
	a_x_half = (float*)malloc(ntx*sizeof(float));    
	a_z      = (float*)malloc(ntz*sizeof(float));
	a_z_half = (float*)malloc(ntz*sizeof(float));


	b_x      = (float*)malloc(ntx*sizeof(float));
	b_x_half = (float*)malloc(ntx*sizeof(float));
	b_z      = (float*)malloc(ntz*sizeof(float));
	b_z_half = (float*)malloc(ntz*sizeof(float));


	k_x      = (float*)malloc(ntx*sizeof(float));
	k_x_half = (float*)malloc(ntx*sizeof(float));
	k_z      = (float*)malloc(ntz*sizeof(float));
	k_z_half = (float*)malloc(ntz*sizeof(float));  

	/*=========================================================
	  Allocate the memory of Seismograms...
	  ========================================================*/
	float tmpfloat;

	for(i=0;i<GPU_N;i++)
	{
		plan[i].seismogram_obs=(float*)malloc(sizeof(float)*itmax*rnmax);
		plan[i].seismogram_dir=(float*)malloc(sizeof(float)*itmax*rnmax);
		plan[i].seismogram_syn=(float*)malloc(sizeof(float)*itmax*rnmax);
		plan[i].seismogram_rms=(float*)malloc(sizeof(float)*itmax*rnmax);
		plan[i].seismogram_rmsup=(float*)malloc(sizeof(float)*itmax*rnmax);
		plan[i].seismogram_rmsdown=(float*)malloc(sizeof(float)*itmax*rnmax);
		plan[i].seismogram_frms=(float*)malloc(sizeof(float)*itmax*rnmax);
/*
		plan[i].p_borders_up    =(float*)malloc(sizeof(float)*2*Lc*itmax*nx);
		plan[i].p_borders_bottom=(float*)malloc(sizeof(float)*2*Lc*itmax*nx);
		plan[i].p_borders_left  =(float*)malloc(sizeof(float)*2*Lc*itmax*(nz-4*Lc));
		plan[i].p_borders_right =(float*)malloc(sizeof(float)*2*Lc*itmax*(nz-4*Lc));
*/
	}

	/*=========================================================
	  Allocate the memory of image / gradient...
	  ========================================================*/

	for(i=0;i<GPU_N;i++)
	{
		plan[i].image_pp=(float*)malloc(sizeof(float)*ntp);
		plan[i].image_ppcig=(float*)malloc(sizeof(float)*ntp*thetan);

		plan[i].image_sources=(float*)malloc(sizeof(float)*ntp);
		plan[i].image_receivers=(float*)malloc(sizeof(float)*ntp);
	}

	tmp1=(float*)malloc(sizeof(float)*np);
	tmp2=(float*)malloc(sizeof(float)*np*thetan);
	tmp3=(float*)malloc(sizeof(float)*np);

	migration_pp=(float*)malloc(sizeof(float)*np);
	migration_ppt=(float*)malloc(sizeof(float)*np);
	migration_ppcig=(float*)malloc(sizeof(float)*np*thetan);

	////////============================////////
	variables_malloc(ntx, ntz, ntp, nx, nz,
		pml, Lc, dx, dz, itmax, thetan, delt,
		plan, GPU_N, rnmax
		);

	/*=========================================================
	  Calculate the ricker wave...
	  ========================================================*/

	if(myid==0)
	{
		ricker_wave(rick,itmax,f0,t0,dt,2);
		printf("Ricker wave is done\n");
	}

	MPI_Barrier(comm);
	MPI_Bcast(rick,itmax,MPI_FLOAT,0,comm);

	//for(i=0;i<GPU_N;i++)
	//	hilbert_transform1d(rick, hrick, i, itmax);

	/*=========================================================
	  Calculate the ture model.../Or read in the true model
	  ========================================================*/

	memset(vp,0,sizeof(float)*ntp);
	if(myid==0)
	{
		get_acc_model(vp,rho,ntp,ntx,ntz,pmlc,v_file);

		fp=fopen("./output/acc_vp.bin","wb");
		for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
		{
			for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
			{
				fwrite(&vp[ix*ntz+iz],sizeof(float),1,fp);

			}
		}
		fclose(fp);

		fp=fopen("./output/acc_rho.bin","wb");
		for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
		{
			for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
			{
				fwrite(&rho[ix*ntz+iz],sizeof(float),1,fp);

			}
		}
		fclose(fp);

		printf("The true model is done\n"); 

	}//end myid

	MPI_Barrier(comm);
	MPI_Bcast(vp,ntp,MPI_FLOAT,0,comm);
	MPI_Bcast(rho,ntp,MPI_FLOAT,0,comm);

	/////////////////////////////////////////////////////
	vp_max=0.0;
	rho_max=0.0;
	vp_min=5000.0;
	rho_min=5000.0;
	for(ip=0;ip<ntp;ip++)
	{     
		if(vp[ip]>=vp_max)
		{
			vp_max=vp[ip];
		}
		if(rho[ip]>=rho_max)
		{
			rho_max=fabs(rho[ip]);
		}
		if(vp[ip]<=vp_min)
		{
			vp_min=vp[ip];
		}
		if(rho[ip]<=rho_min)
		{
			rho_min=fabs(rho[ip]);
		}
	}
	if(myid==0)
	{
		printf("vp_max = %f\n",vp_max); 
		printf("rho_max = %f\n",rho_max);

		printf("vp_min = %f\n",vp_min); 
		printf("rho_min = %f\n",rho_min);
	}

	ini_model_mine(vp,vpn,ntp,ntz,ntx,pmlc,window);

	if(myid==0)
	{
		fp=fopen("./output/ini_vp.bin","wb");
		for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
		{
			for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
			{
				fwrite(&vp[ix*ntz+iz],sizeof(float),1,fp);

			}
		}
		fclose(fp);

		printf("The initial model is done\n"); 
	}//end myid
	/*=========================================================
	  Calculate the parameters of absorbing layers...
	  ========================================================*/

	get_absorbing_parameters(
			d_x,d_x_half,d_z,d_z_half,
			a_x,a_x_half,a_z,a_z_half,
			b_x,b_x_half,b_z,b_z_half,
			k_x,k_x_half,k_z,k_z_half,
			ntz,ntx,nz,nx,pmlc,dx,dz,f0,t0,
			dt,vp_max
			);

	if(myid==0)
		printf("ABC parameters are done\n");

	nsid=ns/(GPU_N*numprocs);
	modsr=ns%(GPU_N*numprocs);
	prcs=modsr/GPU_N;
	prcss=modsr%GPU_N;

	if(prcss==0)
	{
		if(myid<prcs)
		{
			eachsid=nsid+1;
			offsets=myid*(nsid+1)*GPU_N;
		}
		else
		{
			eachsid=nsid;
			offsets=prcs*(nsid+1)*GPU_N+(myid-prcs)*nsid*GPU_N;
		}
	}
	else
	{
		if(myid<=prcs)
		{
			eachsid=nsid+1;
			offsets=myid*(nsid+1)*GPU_N;
		}
		else
		{
			eachsid=nsid;
			offsets=prcs*(nsid+1)*GPU_N+prcss*(nsid+1)+(GPU_N-prcss)*nsid+
				(myid-prcs-1)*nsid*GPU_N;
		}
	}

	if(myid==0)
	{
		start=clock();
	}

	/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	  !	        ITERATION OF FWI IN TIME DOMAIN BEGINS...                      !
	  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

	float alpharms;

	/*=======================================================
	  Back-propagate the RMS wavefields and Construct 
	  the forward wavefield..Meanwhile the gradients 
	  of lambda and mu are computed... 
	  ========================================================*/
	if(myid==0)
	{
		printf("====================\n");
		if(scatter_flag)
			printf(" PSF RTM BEGIN\n");
		else
			printf(" NORMAL RTM BEGIN\n");
		printf("====================\n");
	}

	inv_flag=1; //     FWI   FLAG
	
	float remax,pmax;
	int itheta;

	for(ip=0;ip<np;ip++)
	{
		tmp1[ip]=0.0;
		tmp3[ip]=0.0;
		migration_pp[ip]=0.0;
		migration_ppt[ip]=0.0;
	}
	for(ip=0;ip<np*thetan;ip++)
	{
		tmp2[ip]=0.0;
		migration_ppcig[ip]=0.0;
	}

	/********** FORWARD & BACKWARD ***********/
	for(iss=0;iss<eachsid;iss++)
	{
		is=offsets+iss*GPU_N;

		if(prcss!=0&&myid==prcs&&iss==eachsid-1)
			GPU_N=prcss;
		else
			GPU_N=GPU_NN;

		for(i=0;i<GPU_N;i++)
		{
			for(ip=0;ip<ntp;ip++)
			{
				plan[i].image_pp[ip]=0.0;

				plan[i].image_sources[ip]=0.0;
				plan[i].image_receivers[ip]=0.0;
			}
			for(ip=0;ip<ntp*thetan;ip++)
			{
				plan[i].image_ppcig[ip]=0.0;
			}
		}

		fdtd_2d_GPU_forward(ntx,ntz,ntp,nx,nz,pml, Lc, rc,dx,dz,
				rick,itmax,dt,myid,
				is, ss, plan, GPU_N, rnmax,
				vp,rho,mx,
				k_x,k_x_half,k_z,k_z_half,
				a_x,a_x_half,a_z,a_z_half,
				b_x,b_x_half,b_z,b_z_half,c,
				inv_flag
				);

		// READ IN OBSERVED SEISMOGRAMS...  

		for(i=0;i<GPU_N;i++)
		{
			//if(scatter_flag)
			//	sprintf(filename,"./output/%d%s",is+i+1,scatter_file);
			//else
			//	sprintf(filename,"./output/%d%s",is+i+1,data_file);
			if(scatter_flag)
				sprintf(filename,"./output/%dsource_seismogram_scatterup.bin",is+i+1);
			else
				sprintf(filename,"./output/%dsource_seismogram_obsup.bin",is+i+1);
			if(NULL==(fp=fopen(filename,"rb")))
			{
				printf("Cannot Open the File: %s\n",filename);
				return(0);
			}
			fread(&plan[i].seismogram_frms[0],sizeof(float),ss[is+i].r_n*itmax,fp);
			fclose(fp);
			
			//////// preprocessing the backward residuals //////
			//rmsgpu_fre(plan[i].seismogram_rms,itmax,ss[is+i].r_n,i,dt);
			memset(plan[i].seismogram_rms,0.0,sizeof(float)*ss[is+i].r_n*itmax);
			for(ix=0;ix<ss[is+i].r_n;ix++)
			{
				tmpfloat=0.0;
				for(it=0;it<itmax;it++)
				{
					ip=ix*itmax+it;
					ipp=ix*itmax+it-t0n;

					tmpfloat+=plan[i].seismogram_frms[ip]*dt;

					//if(it-t0n>=0)
					plan[i].seismogram_frms[ip]=tmpfloat;
				}
			}
			for(ix=0;ix<ss[is+i].r_n;ix++)
			{
				tmpfloat=0.0;
				for(it=0;it<itmax;it++)
				{
					ip=ix*itmax+it;
					ipp=ix*itmax+it+t0n;

					tmpfloat+=plan[i].seismogram_frms[ip]*dt;

					//plan[i].seismogram_rms[ip]=-tmpfloat;
					if(it+t0n<itmax)
						plan[i].seismogram_rms[ipp]=-tmpfloat;
				}
			}

			//// output the seismogram ////
			if(is%20==0)
			{
				sprintf(filename,"./output/%dsource_seismogram_syn.bin",is+i+1);
				fp=fopen(filename,"wb");
				fwrite(&plan[i].seismogram_syn[0],sizeof(float),ss[is+i].r_n*itmax,fp);
				fclose(fp);

				sprintf(filename,"./output/%dsource_seismogram_frms.bin",is+i+1);
				fp=fopen(filename,"wb");
				fwrite(&plan[i].seismogram_rms[0],sizeof(float),ss[is+i].r_n*itmax,fp);
				fclose(fp);
			}
		}//end GPU

		fdtd_2d_GPU_backward(ntx,ntz,ntp,nx,nz,pml, Lc, rc,dx,dz,
				rick,itmax,dt, myid, thetan, delt,
				is, ss, plan, GPU_N, rnmax,
				vp,rho,mx,
				k_x,k_x_half,k_z,k_z_half,
				a_x,a_x_half,a_z,a_z_half,
				b_x,b_x_half,b_z,b_z_half
				);

		for(i=0;i<GPU_N;i++)
		{
			//normalize image
			remax=0;
			for(ip=0;ip<ntp;ip++)
			{
				if(remax<fabs(plan[i].image_sources[ip]))
				{
					remax=fabs(plan[i].image_sources[ip]);
				}
			}

			//normalize 
			for(ip=0;ip<ntp;ip++)
			{
				plan[i].image_pp[ip]/=(plan[i].image_sources[ip]+1.0e-3*remax);
			}

			/*pmax=0;
			for(ip=0;ip<ntp;ip++)
			{
				if(pmax<fabs(plan[i].image_pp[ip]))
				{
					pmax=fabs(plan[i].image_pp[ip]);
				}
			}

			for(ip=0;ip<ntp;ip++)
			{
				plan[i].image_pp[ip]/=(pmax);
			}
*/
			if(scatter_flag)
				sprintf(filename,"./output/image_psf%d.bin",is+i+1);
			else
				sprintf(filename,"./output/image_pp%d.bin",is+i+1);
			fp=fopen(filename,"wb");
			for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
			{
				for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
				{
					fwrite(&plan[i].image_pp[ix*ntz+iz],sizeof(float),1,fp);

				}
			}
			fclose(fp);

			sprintf(filename,"./output/image_sources%d.bin",is+i+1);
			fp=fopen(filename,"wb");
			for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
			{
				for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
				{
					fwrite(&plan[i].image_sources[ix*ntz+iz],sizeof(float),1,fp);

				}
			}
			fclose(fp);

			//laplace Filter
			for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
			{
				for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
				{
					ip=ix*ntz+iz;
					ipp=(ix-pmlc)*nz+iz-pmlc;

					tmp3[ipp]=-((plan[i].image_pp[ip+ntz]-2.0*plan[i].image_pp[ip]+plan[i].image_pp[ip-ntz])/(dx*dx)
							+(plan[i].image_pp[ip+1]-2.0*plan[i].image_pp[ip]+plan[i].image_pp[ip-1])/(dz*dz))*vp[ip]*vp[ip];   // inner gradient...        

					tmp1[ipp]+=tmp3[ipp];   // inner gradient...  
				}//ix
			}//iz

			if(scatter_flag)
				sprintf(filename,"./output/fimage_psf%d.bin",is+i+1);
			else
				sprintf(filename,"./output/fimage_pp%d.bin",is+i+1);
			fp=fopen(filename,"wb");
			fwrite(&tmp3[0],sizeof(float),np,fp);
			fclose(fp);
				
		}//end GPU
	}//end is (shotnumbers)

	MPI_Barrier(comm);

	MPI_Allreduce(tmp1,migration_pp,np,MPI_FLOAT,MPI_SUM,comm);
	//MPI_Allreduce(tmp2,migration_ppcig,np*thetan,MPI_FLOAT,MPI_SUM,comm);
	//MPI_Allreduce(tmp3,migration_ppt,np,MPI_FLOAT,MPI_SUM,comm);

	for(ix=0;ix<nx;ix++)
	{
		for(iz=0;iz<nz;iz++)
		{
			ip=ix*nz+iz;

			if(ix<=3||ix>=nx-Lc||iz<7||iz>=nz-Lc)
				migration_pp[ip]=0.0;//migration_pp[2*nz+iz];
			/*if(ix>=nx-3)
				migration_pp[ip]=migration_pp[(nx-3)*nz+iz];
			if(iz>=nz-3)
				migration_pp[ip]=migration_pp[ix*nz+nz-3];
				*/
		}
	}

	/*==========================================================
	  Output the updated model such as vp,rho,...
	  ===========================================================*/

	if(myid==0)
	{
		if(scatter_flag)
			sprintf(filename,"./output/migration_psf.bin");
		else
			sprintf(filename,"./output/migration_pp.bin");
		fp=fopen(filename,"wb");
		fwrite(&migration_pp[0],sizeof(float),np,fp);
		fclose(fp);

		/*sprintf(filename,"./output/migration_ppcig.bin");
		fp=fopen(filename,"wb");
		fwrite(&migration_ppcig[0],sizeof(float),np*thetan,fp);
		fclose(fp);

		sprintf(filename,"./output/migration_ppt.bin");
		fp=fopen(filename,"wb");
		fwrite(&migration_ppt[0],sizeof(float),np,fp);
		fclose(fp);*/
	}

	MPI_Barrier(comm);

	if(myid==0)
	{
		printf("====================\n");
		printf("      THE END\n");
		printf("====================\n");

		end=clock();
		printf("The cost of the run time is %f seconds\n",
				(double)(end-start)/CLOCKS_PER_SEC);
	}
	/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	  !	        ITERATION OF FWI IN TIME DOMAIN ENDS...                        !
	  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

	variables_free(ntx, ntz, ntp, nx, nz,
			pml, Lc, dx, dz, itmax,
			plan, GPU_N, rnmax
			);

	free(rick);
	free(hrick);
	free(rc); 

	for(is=0;is<ns;is++)
	{
		free(ss[is].r_iz);
	} 

	free(a_x);
	free(a_x_half);
	free(a_z);
	free(a_z_half);

	free(b_x);
	free(b_x_half);
	free(b_z);
	free(b_z_half);

	free(d_x);
	free(d_x_half);
	free(d_z);
	free(d_z_half);

	free(k_x);
	free(k_x_half);
	free(k_z);
	free(k_z_half);

	//free the memory of P velocity
	free(vp);
	//free the memory of Density
	free(rho); 
	//free the memory of lamda+2Mu

	free(vpn); 
	free(rhon);

	for(i=0;i<GPU_N;i++)
	{
		free(plan[i].seismogram_obs);
		free(plan[i].seismogram_dir);
		free(plan[i].seismogram_syn); 
		free(plan[i].seismogram_rms);
		free(plan[i].seismogram_frms);
		/*
		   free(plan[i].p_borders_up);
		   free(plan[i].p_borders_bottom);
		   free(plan[i].p_borders_left);
		   free(plan[i].p_borders_right);
		 */
		free(plan[i].image_pp);
		free(plan[i].image_ppcig);
		free(plan[i].image_sources);
		free(plan[i].image_receivers);
	}

	free(tmp1);
	free(tmp2);
	free(tmp3);

	free(migration_pp);
	free(migration_ppt);
	free(migration_ppcig);

	MPI_Barrier(comm);
	MPI_Finalize();
}


/*==========================================================
  This subroutine is used for calculating the parameters of 
  absorbing layers
  ===========================================================*/

void get_absorbing_parameters(
		float *d_x, float *d_x_half, 
		float *d_z, float *d_z_half,
		float *a_x, float *a_x_half,
		float *a_z, float *a_z_half,
		float *b_x, float *b_x_half,
		float *b_z, float *b_z_half,
		float *k_x, float *k_x_half,
		float *k_z, float *k_z_half,
		int ntz, int ntx, int nz, int nx,
		int pml, float dx, float dz, float f0, float t0, float dt, float vp_max)
{
	int   N=2;
	int   iz,ix;

	float thickness_of_pml;
	float Rc=1.0e-5;

	float d0;
	float pi=3.1415927;
	float alpha_max=pi*15;

	float Vpmax;


	float *alpha_x,*alpha_x_half;
	float *alpha_z,*alpha_z_half;

	float x_start,x_end,delta_x;
	float z_start,z_end,delta_z;
	float x_current,z_current;

	Vpmax=5500;

	thickness_of_pml=pml*dz;

	d0=-(N+1)*Vpmax*log(Rc)/(2.0*thickness_of_pml);

	alpha_x      = (float*)malloc(ntx*sizeof(float));
	alpha_x_half = (float*)malloc(ntx*sizeof(float));

	alpha_z      = (float*)malloc(ntz*sizeof(float));
	alpha_z_half = (float*)malloc(ntz*sizeof(float));

	//--------------------initialize the vectors--------------

	for(ix=0;ix<ntx;ix++)
	{
		a_x[ix]          = 0.0;
		a_x_half[ix]     = 0.0;
		b_x[ix]          = 0.0;
		b_x_half[ix]     = 0.0;
		d_x[ix]          = 0.0;
		d_x_half[ix]     = 0.0;
		k_x[ix]          = 1.0;
		k_x_half[ix]     = 1.0;
		alpha_x[ix]      = 0.0;
		alpha_x_half[ix] = 0.0;
	}

	for(iz=0;iz<ntz;iz++)
	{
		a_z[iz]          = 0.0;
		a_z_half[iz]     = 0.0;
		b_z[iz]          = 0.0;
		b_z_half[iz]     = 0.0;
		d_z[iz]          = 0.0;
		d_z_half[iz]     = 0.0;
		k_z[iz]          = 1.0;
		k_z_half[iz]     = 1.0;

		alpha_z[iz]      = 0.0;
		alpha_z_half[iz] = 0.0;
	}


	// X direction

	x_start=pml*dx;
	x_end=(ntx-pml-1)*dx;

	// Integer points
	for(ix=0;ix<ntx;ix++)
	{ 
		x_current=ix*dx;

		// LEFT EDGE
		if(x_current<=x_start)
		{
			delta_x=x_start-x_current;
			d_x[ix]=d0*pow(delta_x/thickness_of_pml,2);
			k_x[ix]=1.0;
			alpha_x[ix]=alpha_max*(1.0-(delta_x/thickness_of_pml))+0.1*alpha_max;
		}

		// RIGHT EDGE      
		if(x_current>=x_end)
		{
			delta_x=x_current-x_end;
			d_x[ix]=d0*pow(delta_x/thickness_of_pml,2);
			k_x[ix]=1.0;
			alpha_x[ix]=alpha_max*(1.0-(delta_x/thickness_of_pml))+0.1*alpha_max;
		}
	}


	// Half Integer points
	for(ix=0;ix<ntx;ix++)
	{
		x_current=(ix+0.5)*dx;

		if(x_current<=x_start)
		{
			delta_x=x_start-x_current;
			d_x_half[ix]=d0*pow(delta_x/thickness_of_pml,2);
			k_x_half[ix]=1.0;
			alpha_x_half[ix]=alpha_max*(1.0-(delta_x/thickness_of_pml))+0.1*alpha_max;
		}

		if(x_current>=x_end)
		{
			delta_x=x_current-x_end;
			d_x_half[ix]=d0*pow(delta_x/thickness_of_pml,2);
			k_x_half[ix]=1.0;
			alpha_x_half[ix]=alpha_max*(1.0-(delta_x/thickness_of_pml))+0.1*alpha_max;
		}
	}

	for (ix=0;ix<ntx;ix++)
	{
		if(alpha_x[ix]<0.0)
		{
			alpha_x[ix]=0.0;
		}
		if(alpha_x_half[ix]<0.0)
		{
			alpha_x_half[ix]=0.0;
		}

		b_x[ix]=exp(-(d_x[ix]/k_x[ix]+alpha_x[ix])*dt);

		if(d_x[ix] > 1.0e-6)
		{
			a_x[ix]=d_x[ix]/(k_x[ix]*(d_x[ix]+k_x[ix]*alpha_x[ix]))*(b_x[ix]-1.0);
		}

		b_x_half[ix]=exp(-(d_x_half[ix]/k_x_half[ix]+alpha_x_half[ix])*dt);

		if(d_x_half[ix] > 1.0e-6)
		{
			a_x_half[ix]=d_x_half[ix]/(k_x_half[ix]*(d_x_half[ix]+k_x_half[ix]*alpha_x_half[ix]))*(b_x_half[ix]-1.0);
		}
	}

	// Z direction

	z_start=pml*dz;
	z_end=(ntz-pml-1)*dz;

	// Integer points
	for(iz=0;iz<ntz;iz++)
	{ 
		z_current=iz*dz;

		// LEFT EDGE
		if(z_current<=z_start)
		{
			delta_z=z_start-z_current;
			d_z[iz]=d0*pow(delta_z/thickness_of_pml,2);
			k_z[iz]=1.0;
			alpha_z[iz]=alpha_max*(1.0-(delta_z/thickness_of_pml))+0.1*alpha_max;
		}

		// RIGHT EDGE      
		if(z_current>=z_end)
		{
			delta_z=z_current-z_end;
			d_z[iz]=d0*pow(delta_z/thickness_of_pml,2);
			k_z[iz]=1.0;
			alpha_z[iz]=alpha_max*(1.0-(delta_z/thickness_of_pml))+0.1*alpha_max;
		}
	}

	// Half Integer points
	for(iz=0;iz<ntz;iz++)
	{
		z_current=(iz+0.5)*dz;

		if(z_current<=z_start)
		{
			delta_z=z_start-z_current;
			d_z_half[iz]=d0*pow(delta_z/thickness_of_pml,2);
			k_z_half[iz]=1.0;
			alpha_z_half[iz]=alpha_max*(1.0-(delta_z/thickness_of_pml))+0.1*alpha_max;
		}

		if(z_current>=z_end)
		{
			delta_z=z_current-z_end;
			d_z_half[iz]=d0*pow(delta_z/thickness_of_pml,2);
			k_z_half[iz]=1.0;
			alpha_z_half[iz]=alpha_max*(1.0-(delta_z/thickness_of_pml))+0.1*alpha_max;
		}
	}

	for (iz=0;iz<ntz;iz++)
	{
		if(alpha_z[iz]<0.0)
		{
			alpha_z[iz]=0.0;
		}
		if(alpha_z_half[iz]<0.0)
		{
			alpha_z_half[iz]=0.0;
		}

		b_z[iz]=exp(-(d_z[iz]/k_z[iz]+alpha_z[iz])*dt);

		if(d_z[iz]>1.0e-6)
		{
			a_z[iz]=d_z[iz]/(k_z[iz]*(d_z[iz]+k_z[iz]*alpha_z[iz]))*(b_z[iz]-1.0);
		}

		b_z_half[iz]=exp(-(d_z_half[iz]/k_z_half[iz]+alpha_z_half[iz])*dt);

		if(d_z_half[iz]>1.0e-6)
		{
			a_z_half[iz]=d_z_half[iz]/(k_z_half[iz]*(d_z_half[iz]+k_z_half[iz]*alpha_z_half[iz]))*(b_z_half[iz]-1.0);
		}
	}

	free(alpha_x);
	free(alpha_x_half);
	free(alpha_z);
	free(alpha_z_half);

	return;

}


/*==========================================================
  This subroutine is used for initializing the true model...
  ===========================================================*/

void get_acc_model(float *vp, float *rho, int ntp, int ntx, int ntz, int pml, char *v_file)
{
	int ip,ipp,iz,ix;
	// THE MODEL    
	FILE *fp;
	char filename[100];

	//fp=fopen("./input/acc_vp.bin","rb");
	//fp=fopen("./input/vel_ore_721x1601_5x5.bin","rb");
	sprintf(filename,"./input/acc_vp.bin");
	fp=fopen(filename,"rb");
	for(ix=pml;ix<ntx-pml;ix++)
	{
		for(iz=pml;iz<ntz-pml;iz++)
		{
			ip=ix*ntz+iz;
			fread(&vp[ip],sizeof(float),1,fp);           
		}
	}
	fclose(fp);

	// Model in PML
	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=0;ix<=pml-1;ix++)
		{
			ip=ix*ntz+iz;
			ipp=pml*ntz+iz;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=ix*ntz+iz;
			ipp=(ntx-pml-1)*ntz+iz;

			vp[ip]=vp[ipp];
		}

	}

	for(ix=0;ix<ntx;ix++)
	{
		for(iz=0;iz<=pml-1;iz++)
		{
			ip=ix*ntz+iz;
			ipp=ix*ntz+pml;

			vp[ip]=vp[ipp];
		}

		for(iz=ntz-pml;iz<ntz;iz++)
		{
			ip=ix*ntz+iz;
			ipp=ix*ntz+(ntz-pml-1);

			vp[ip]=vp[ipp];
		}
	}

	///////////
	for(ip=0;ip<ntp;ip++)
	{
		rho[ip]=1000.0;
	}
		
	return;
}


/*==========================================================
  This subroutine is used for finding the maximum value of 
  a vector.
  ===========================================================*/ 
void maximum_vector(float *vector, int n, float *maximum_value)
{
	int i;

	*maximum_value=1.0e-20;
	for(i=0;i<n;i++)
	{
		if(vector[i]>*maximum_value);
		{
			*maximum_value=vector[i];
		}
	}
	printf("maximum_value=%f\n",*maximum_value);
	return;
}


/*==========================================================
  This subroutine is used for calculating the sum of two 
  vectors!
  ===========================================================*/

void add(float *a,float *b,float *c,int n)
{
	int i;
	for(i=0;i<n;i++)
	{
		c[i]=a[i]-b[i];
	}

}

/*==========================================================

  This subroutine is used for calculating the ricker wave

  ===========================================================*/

void ricker_wave(float *rick, int itmax, float f0, float t0, float dt, int flag)
{
	float pi=3.1415927;
	int   it;
	float temp,max=0.0;

	FILE *fp;

	if(flag==3)
	{	
		for(it=0;it<itmax;it++)
		{
			temp=1.5*pi*f0*(it*dt-t0);
			temp=temp*temp;
			rick[it]=exp(-temp);  

			if(max<fabs(rick[it]))
			{
				max=fabs(rick[it]);
			}
		}

		for(it=0;it<itmax;it++)
		{
			rick[it]=rick[it]/max;
		}

		fp=fopen("./output/rick_third_derive.bin","wb");    
		for(it=0;it<itmax;it++)
		{
			fwrite(&rick[it],sizeof(float),1,fp);
		}    
		fclose(fp);
	}

	if(flag==2)
	{
		for(it=0;it<itmax;it++)
		{
			temp=pi*f0*(it*dt-t0);
			temp=temp*temp;
			rick[it]=(1.0-2.0*temp)*exp(-temp);
		}

		fp=fopen("./output/rick_second_derive.bin","wb");    
		for(it=0;it<itmax;it++)
		{
			fwrite(&rick[it],sizeof(float),1,fp);
		}    
		fclose(fp);
	}
	if(flag==1)
	{
		for(it=0;it<itmax;it++)
		{
			temp=pi*f0*(it*dt-t0);
			temp=temp*temp;         
			rick[it]=(it*dt-t0)*exp(-temp);

			if(max<fabs(rick[it]))
			{
				max=fabs(rick[it]);
			}
		}

		for(it=0;it<itmax;it++)
		{
			rick[it]=rick[it]/max;
		}

		fp=fopen("./output/rick_first_derive.bin","wb");    
		for(it=0;it<itmax;it++)
		{
			fwrite(&rick[it],sizeof(float),1,fp);
		}    
		fclose(fp);
	}

	return;
}

//*************************************************************************
//*******un0*cnmax=vmax*0.01
//************************************************************************

void ini_step(float *dn, int np, float *un0, float max)
{
	float dnmax=0.0;
	int ip;

	for(ip=0;ip<np;ip++)
	{
		if(dnmax<fabs(dn[ip]))
		{
			dnmax=fabs(dn[ip]);
		}
	}   

	*un0=max*0.01/dnmax;    

	return;
}


/*=========================================================================
  To calculate the updated model...
  ========================================================================*/

void update_model(float *vp, float *vpn,
		float *dn_vp, float *un_vp,
		int ntp, int ntz, int ntx, int pml, float vpmin, float vpmax)
{
	int ip,ipp;
	int iz,ix;
	int nz=ntz-2*pml;

	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=ix*ntz+iz;
			ipp=(ix-pml)*nz+iz-pml;
			vp[ip]=vpn[ip]+*un_vp*dn_vp[ipp];

			if(vp[ip]<0.8*vpmin)
				vp[ip]=0.8*vpmin;
			if(vp[ip]>1.2*vpmax)
				vp[ip]=1.2*vpmax;
		}
	}

	//  Model in PML..............

	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=0;ix<=pml-1;ix++)
		{
			ip=ix*ntz+iz;
			ipp=pml*ntz+iz;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=ix*ntz+iz;
			ipp=(ntx-pml-1)*ntz+iz;

			vp[ip]=vp[ipp];
		}

	}

	for(ix=0;ix<ntx;ix++)
	{
		for(iz=0;iz<=pml-1;iz++)
		{
			ip=ix*ntz+iz;
			ipp=ix*ntz+pml;

			vp[ip]=vp[ipp];
		}

		for(iz=ntz-pml;iz<ntz;iz++)
		{
			ip=ix*ntz+iz;
			ipp=ix*ntz+(ntz-pml-1);

			vp[ip]=vp[ipp];
		}
	}
	return;
}


/*==========================================================
  This subroutine is used for expand the model to PML area...
  ===========================================================*/

void model_in_pml(float *vp, int ntp, int ntz, int ntx, int pml)
{
	int iz,ix;
	int ip,ipp;

	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=0;ix<=pml-1;ix++)
		{
			ip=ix*ntz+iz;
			ipp=pml*ntz+iz;
			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=ix*ntz+iz;
			ipp=(ntx-pml-1)*ntz+iz;

			vp[ip]=vp[ipp];
		}

	}
	for(ix=0;ix<ntx;ix++)
	{
		for(iz=0;iz<=pml-1;iz++)
		{
			ip=ix*ntz+iz;
			ipp=ix*ntz+pml;

			vp[ip]=vp[ipp];
		}
		for(iz=ntz-pml;iz<ntz;iz++)
		{
			ip=ix*ntz+iz;
			ipp=ix*ntz+(ntz-pml-1);

			vp[ip]=vp[ipp];
		}
	}
}

/***********************************************************************
  !                initial model
  !***********************************************************************/
void ini_model_mine(float *vp, float *vp_n, int ntp, int ntz, int ntx, int pml, int window)
{
	int windxz=2*window+1;
	double sigma=window/2.0;

	double sum;
	int number;

	int iz,ix;
	int izw,ixw,iz1,ix1;
	int ip,ip1,ipp;

	/////////////////////////////////////////
	double csum=0.0;
	double *coef=(double *)malloc(sizeof(double)*windxz*windxz);

	sum=0.0;
	for(ix=0;ix<windxz;ix++)
		for(iz=0;iz<windxz;iz++)
			{
				ip=ix*windxz+iz;
				coef[ip]=1.0/(2.0*PI*sigma*sigma)*exp(-((ix-window)*(ix-window)+(iz-window)*(iz-window))/(2.0*sigma*sigma));
				sum+=coef[ip];
			}
	for(ip=0;ip<windxz*windxz;ip++)
	{
		coef[ip]/=sum;
		csum+=coef[ip];
	}

	/////////////////////////////////////////

	double *vp_old1;

	vp_old1=(double*)malloc(sizeof(double)*ntp);

	for(ip=0;ip<ntp;ip++)
		vp_old1[ip]=vp[ip];

	//-----smooth in the x direction---------

#pragma omp parallel for private(ix,iz,sum,number,ixw,izw,ix1,iz1,ip,ip1,ipp)
	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			sum=0.0;
			number=0;

			for(izw=iz-window;izw<=iz+window;izw++)
			{
				for(ixw=ix-window;ixw<=ix+window;ixw++)
				{
					ipp=(izw-iz+window)*windxz+(ixw-ix+window);

					if(izw<0)
						iz1=0;                		
					else if(izw>ntz-1)
						iz1=ntz-1;
					else
						iz1=izw;

					if(ixw<0)
						ix1=0;
					else if(ixw>ntx-1)
						ix1=ntx-1;
					else
						ix1=ixw;

					ip1=ix1*ntz+iz1;

					sum+=vp_old1[ip1]*coef[ipp];
					number++;
				}
			}
			ip=ix*ntz+iz;
			vp[ip]=sum;///number;

			if(iz<pml+8)
			{
				vp[ip]=vp_old1[ip];
			}
		}
	}    

	//  Model in PML..............

	model_in_pml(vp,ntp,ntz,ntx,pml);

	for(ip=0;ip<ntp;ip++)
	{
		vp_n[ip]=vp[ip];
	}

	free(coef);
	free(vp_old1);
}
void ini_model_minea(float *vp, float *vpn, int ntp, int ntz, int ntx, int pml, int window)
{
	/*  flag == 1 :: P velocity
		flag == 2 :: S velocity
		flag == 3 :: Density
		*/

	float *vp_old1;

	float sum;
	int number;

	int iz,ix;
	int izw,ixw,iz1,ix1;
	int ip,ipp;

	vp_old1=(float*)malloc(sizeof(float)*ntp);


	for(ip=0;ip<ntp;ip++)
	{
		vp_old1[ip]=vp[ip];
	}

	//-----smooth in the x direction---------

	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			sum=0.0;
			number=0;

			for(izw=iz-window;izw<iz+window;izw++)
			{
				for(ixw=ix-window;ixw<ix+window;ixw++)
				{
					if(izw<0)
					{
						iz1=0;                		
					}
					else if(izw>ntz-1)
					{
						iz1=ntz-1;
					}
					else
					{
						iz1=izw;
					}

					if(ixw<0)
					{
						ix1=0;
					}
					else if(ixw>ntx-1)
					{
						ix1=ntx-1;
					}
					else
					{
						ix1=ixw;
					}

					ip=ix1*ntz+iz1;
					sum=sum+vp_old1[ip];
					number=number+1;
				}
			}
			ip=ix*ntz+iz;
			vp[ip]=sum/number;

			if(iz<pml+6)
			{
				vp[ip]=vp_old1[ip];
			}
		}
	}    

	//  Model in PML..............

	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=0;ix<=pml-1;ix++)
		{
			ip=ix*ntz+iz;
			ipp=pml*ntz+iz;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=ix*ntz+iz;
			ipp=(ntx-pml-1)*ntz+iz;

			vp[ip]=vp[ipp];
		}

	}

	for(ix=0;ix<ntx;ix++)
	{
		for(iz=0;iz<=pml-1;iz++)
		{
			ip=ix*ntz+iz;
			ipp=ix*ntz+pml;

			vp[ip]=vp[ipp];
		}

		for(iz=ntz-pml;iz<ntz;iz++)
		{
			ip=ix*ntz+iz;
			ipp=ix*ntz+(ntz-pml-1);

			vp[ip]=vp[ipp];
		}
	}	

	for(ip=0;ip<ntp;ip++)
	{
		vpn[ip]=vp[ip];
	}

	free(vp_old1);

	return;
}


void get_ini_model(float *vp, float *rho, int ntp, int ntx, int ntz, int pml)
{
	int ip,ipp,iz,ix;
	// THE MODEL    
	FILE *fp;

	for(ix=0;ix<ntx;ix++)
	{
		for(iz=0;iz<ntz;iz++)
		{
			ip=ix*ntz+iz;

			if(iz>pml+1)
			{
				ipp=ix*ntz+(pml+1);

				vp[ip]=vp[ipp];
				rho[ip]=rho[ipp];
			}
		}
	}

	return;
}

/*=======================================================================

  subroutine preprocess(nz,nx,dx,dz,P)

  !=======================================================================*/
// in this program Precondition P is computed

void Preprocess(int nz, int nx, float dx, float dz, float *P)
{
	int iz,iz_depth_one,iz_depth_two;
	float z,delta1,a,temp,z1,z2;

	a=3.0;
	iz_depth_one=3;
	iz_depth_two=6;

	delta1=(iz_depth_two-iz_depth_one)*dz;
	z1=(iz_depth_one-1)*dz;
	z2=(iz_depth_two-1)*dz;

	for(iz=0;iz<nz;iz++)
	{ 
		z=iz*dz;
		if(z<=z1)
		{
			P[iz]=0.0;
		}

		if(z>z1&&z<=z2)
		{
			temp=z-z1-delta1;
			temp=a*temp*2/delta1;
			temp=temp*temp;
			P[iz]=exp(-0.5*temp);//0.0;//
		}

		if(z>z2)
		{
			P[iz]=1.0;//float(z)/float(z2)*1.0;//
		}
	}
}

/*===========================================================

  This subroutine is used for FFT/IFFT

  ===========================================================*/
void fft(float *xreal,float *ximag,int n,int sign)
{
	int i,j,k,m,temp;
	int h,q,p;
	float t;
	float *a,*b;
	float *at,*bt;
	int *r;

	a=(float*)malloc(n*sizeof(float));
	b=(float*)malloc(n*sizeof(float));
	r=(int*)malloc(n*sizeof(int));
	at=(float*)malloc(n*sizeof(float));
	bt=(float*)malloc(n*sizeof(float));

	m=(int)(log(n-0.5)/log(2.0))+1; //2的幂，2的m次方等于n；
	for(i=0;i<n;i++)
	{
		a[i]=xreal[i];
		b[i]=ximag[i];
		r[i]=i;
	}
	for(i=0,j=0;i<n-1;i++)  //0到n的反序；
	{
		if(i<j)
		{
			temp=r[i];
			r[i]=j;
			r[j]=temp;
		}
		k=n/2;
		while(k<(j+1))
		{
			j=j-k;
			k=k/2;
		}
		j=j+k;
	}

	t=2*PI/n;
	for(h=m-1;h>=0;h--)
	{
		p=(int)pow(2.0,h);
		q=n/p;
		for(k=0;k<n;k++)
		{
			at[k]=a[k];
			bt[k]=b[k];
		}

		for(k=0;k<n;k++)
		{
			if(k%p==k%(2*p))
			{

				a[k]=at[k]+at[k+p];
				b[k]=bt[k]+bt[k+p];
				a[k+p]=(at[k]-at[k+p])*cos(t*(q/2)*(k%p))-(bt[k]-bt[k+p])*sign*sin(t*(q/2)*(k%p));
				b[k+p]=(bt[k]-bt[k+p])*cos(t*(q/2)*(k%p))+(at[k]-at[k+p])*sign*sin(t*(q/2)*(k%p));
			}
		}

	}

	for(i=0;i<n;i++)
	{
		if(sign==1)
		{
			xreal[r[i]]=a[i];
			ximag[r[i]]=b[i];
		}
		else if(sign==-1)
		{
			xreal[r[i]]=a[i]/n;
			ximag[r[i]]=b[i]/n;
		}
	}

	free(a);
	free(b);
	free(r);
	free(at);
	free(bt);
}

void cal_xishu(int Lx,float *rx)
{
	int m,i;
	float s1,s2;
	for(m=1;m<=Lx;m++)
	{
		s1=1.0;s2=1.0;
		for(i=1;i<m;i++)
		{
			s1=s1*(2.0*i-1)*(2.0*i-1);
			s2=s2*((2.0*m-1)*(2.0*m-1)-(2.0*i-1)*(2.0*i-1));
		}
		for(i=m+1;i<=Lx;i++)
		{
			s1=s1*(2.0*i-1)*(2.0*i-1);
			s2=s2*((2.0*m-1)*(2.0*m-1)-(2.0*i-1)*(2.0*i-1));
		}
		s2=fabs(s2);
		rx[m-1]=pow(-1.0,m+1)*s1/(s2*(2.0*m-1));
	}
}

void ricker_phase_shift(int itmax, float *rick)
{
	int it,zft,nfft;
	zft=(int)ceil(log(1.0*itmax)/log(2.0));
	nfft=(int)pow(2.0,zft);
	int halfitmax=itmax/2;
	int halfft=nfft/2;

	fftw_complex *rick_f=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*nfft);
	fftw_complex *rick_p=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*nfft);

	fftw_plan planfp,planbp;

	planfp=fftw_plan_dft_1d(nfft,rick_f, rick_f, FFTW_FORWARD, FFTW_MEASURE);
	planbp=fftw_plan_dft_1d(nfft,rick_p, rick_p, FFTW_BACKWARD,FFTW_MEASURE);

	for(it=0;it<nfft;it++)
	{
		rick_f[it][0]=0.0;
		rick_f[it][1]=0.0;
		rick_p[it][0]=0.0;
		rick_p[it][1]=0.0;
	}
	for(it=0;it<itmax;it++)
		rick_f[halfft-halfitmax+it][0]=rick[it];

	fftw_execute(planfp);

	for(it=0;it<nfft;it++)
	{
		if(it<round(nfft/2)){
			rick_p[it][0]= rick_f[it][1];
			rick_p[it][1]=-rick_f[it][0];
		}
		else{
			rick_p[it][0]=-rick_f[it][1];
			rick_p[it][1]= rick_f[it][0];
		}
	}
	fftw_execute(planbp);

	for(it=0;it<itmax;it++)
		rick[it]=rick_p[halfft-halfitmax+it][0]/nfft;

	fftw_free(rick_f);
	fftw_free(rick_p);

	fftw_destroy_plan(planfp);
	fftw_destroy_plan(planbp);
}

void input_parameters(int *nx,int *nz,int *pml,int *Lc,float *dx,float *dz,int *itmax,float *dt,float *f0, 
		int *ns,float *sx0,float *shotdx,float *shotdep,
		float *recdix,float *offsetz,int *scatter_flag,int *swin,int *window
		)
{
	char strtmp[256];
	FILE *fp=fopen("./parameter.txt","r");
	if(fp==0)
	{
		printf("Cannot open the parameters1 file!\n");
		exit(0);
	}

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",nx);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",nz);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",pml);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",Lc);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%f",dx);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%f",dz);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",itmax);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp); 
	fscanf(fp,"\n");
	fscanf(fp,"%f",dt);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp); 
	fscanf(fp,"\n");
	fscanf(fp,"%f",f0);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",ns);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%f",sx0);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%f",shotdx);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%f",shotdep);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%f",recdix);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%f",offsetz);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",scatter_flag);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",swin);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",window);
	fscanf(fp,"\n");

	return;
}
/*
	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",r_n);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%f",rx0);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%f",recdx);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%s",v_file);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%s",scatter_file);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%s",data_file);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%s",srn_file);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%s",sx_file);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%s",srx_file);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%s",mx_file);
	fscanf(fp,"\n");
*/
