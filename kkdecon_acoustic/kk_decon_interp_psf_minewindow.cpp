#include "mpi.h"
#include "fftw3.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "omp.h"

#define PI 3.1415926

#include "headpsf.h"

void input_parameters(int *nx,int *nz,int *psfwinx,int *psfwinz,int *filterwinx,int *filterwinz,int *snapwinx,int *snapwinz,float *epsilon,char *vp_file,char *vp0_file,char *rho_file,char *migreal_file,char *migpsf_file,char *decon_file,char *ipsf_file, int *shift_flag,float *reg_eps);
		
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

	FILE *fp;

	//  float runtime=0.0;
	int nx,nz;

	////////////////////////////////////////////////////////
	/////////////             PSF               ////////////
	////////////////////////////////////////////////////////
	int psfwinx,psfwinz;
	int filterwinx,filterwinz;
	int snapwinx,  snapwinz;
	float epsilon;
	int psf_shift_flag;
	float reg_eps;
	int kk_flag=0;
	int svd_flag=0;

	char vp_file[100],vp0_file[100],rho_file[100];
	char migreal_file[100],migpsf_file[100],decon_file[100],ipsf_file[100];

	////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////

	input_parameters(&nx,&nz,&psfwinx,&psfwinz,&filterwinx,&filterwinz,&snapwinx,&snapwinz,&epsilon,vp_file,vp0_file,rho_file,migreal_file,migpsf_file,decon_file,ipsf_file,&psf_shift_flag,&reg_eps);

	if(psfwinx%2==0||psfwinz%2==0
			||filterwinx%2==0||filterwinz%2==0)
	{
		if(myid==0)
			printf("All the PSF and Fitler Windows Should Be Odd Number !\n");
		return(0);
	}
	////////////////////////////////////////////////////////
	/////////////             PSF               ////////////
	////////////////////////////////////////////////////////

	int ix,iz,ip;
	int ixx,izz,ipp,ipc;
	int iix,iiz,iip,iipp;

	int ixf,izf,ixp,izp;

	int halfpsfwx=floor(psfwinx/2);
	int halfpsfwz=floor(psfwinz/2);

	int halfltwx=floor(filterwinx/2);
	int halfltwz=floor(filterwinz/2);

	int np =nx*nz;

	int nzwpsf=ceil(1.0*nz/psfwinz);
	int nxwpsf=ceil(1.0*nx/psfwinx);

	//////////////  full win  //////////
	int fullwx=filterwinx+psfwinx-1;//(filterwinx>psfwinx?filterwinx:psfwinx);
	int fullwz=filterwinz+psfwinz-1;//(filterwinz>psfwinz?filterwinz:psfwinz);

	int halffwx=fullwx/2;
	int halffwz=fullwz/2;
	
	int midpx=halffwx+1-1; // begin from 0
	int midpz=halffwz+1-1; // begin from 0

	int outwinx=2*(psfwinx/10)+2*snapwinx+1;//realpsfwinx+snapwinx;
	int outwinz=2*(psfwinz/10)+2*snapwinz+1;//realpsfwinz+snapwinz;

	int halfoutwx=floor(outwinx/2);
	int halfoutwz=floor(outwinz/2);

	int midoutwx=halfoutwx+1-1;
	int midoutwz=halfoutwz+1-1;

	int stepwinx=outwinx-snapwinx;
	int stepwinz=outwinz-snapwinz;

	int stepnumx=ceil(1.0*(nx+fullwx)/stepwinx)+1;
	int stepnumz=ceil(1.0*(nz+fullwz)/stepwinz)+1;
	int totalnum=stepnumx*stepnumz;

	int nxp=stepnumx*stepwinx;
	int nzp=stepnumz*stepwinz;

	int npf=nxp*nzp;

	if(filterwinx<psfwinx || filterwinz<psfwinz)
	{printf("Filter length < psf's window length !\nBreak up!\n");return(0);}
	if(filterwinx<outwinx || filterwinz<outwinz)
	{printf("Filter length < output window length !\nBreak up!\n");return(0);}

	if(myid==0)
		printf("Full window size -- (%d,  %d)\nFilter window size -- (%d,  %d)\nOutput size -- (%d,  %d)\nStep window size -- (%d,  %d)\n",fullwz,fullwx,filterwinz,filterwinx,outwinz,outwinx,stepwinx,stepwinz);
	
	/*-------------------------------------------------------------*/

	/*=========================================================
	  Parameters of model...
	  ========================================================*/

	float *vp,*rho,*mn;
	float *vp_old;
	float *vpf;
	float *migreal, *migpsf;
	float *migrealf, *migpsff, *migpsffl;
	float *migcoefp;
	float *psf_inv,*mig_dec;
	float *mig_coef;
	float *coefr,*coefp;
	float *afft_max;
	float mn_max,vp_max;
	float obsmax,synmax;

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
	  Allocate the memory of parameters of model...
	  ========================================================*/

	// allocate the memory of model parameters...

	vp               = (float*)malloc(sizeof(float)*np);
	rho              = (float*)malloc(sizeof(float)*np);

	mn               = (float*)malloc(sizeof(float)*np);

	vp_old  = (float*)malloc(sizeof(float)*np);

	migreal = (float*)malloc(sizeof(float)*np);
	migpsf  = (float*)malloc(sizeof(float)*np);

	vpf     = (float*)malloc(sizeof(float)*npf);

	migrealf= (float*)malloc(sizeof(float)*npf);
	migpsff = (float*)malloc(sizeof(float)*npf);

	migpsffl= (float*)malloc(sizeof(float)*npf);
	migcoefp= (float*)malloc(sizeof(float)*npf);

	mig_dec = (float*)malloc(sizeof(float)*npf);
	mig_coef= (float*)malloc(sizeof(float)*npf);
	psf_inv = (float*)malloc(sizeof(float)*npf);

	coefr   =(float*)malloc(sizeof(float)*outwinx*outwinz);
	coefp   =(float*)malloc(sizeof(float)*psfwinx*psfwinz);
	afft_max=(float*)malloc(sizeof(float)*stepnumx*stepnumz);

	/*=========================================================
	  Calculate the ture model.../Or read in the true model
	  ========================================================*/

	if(myid==0)
	{
		fp=fopen(vp_file,"rb");
		fread(&vp_old[0],sizeof(float),np,fp);
		fclose(fp);

		fp=fopen("./output/acc_vp.bin","wb");
		fwrite(&vp_old[0],sizeof(float),np,fp);
		fclose(fp);

		memset(mn,0.0,sizeof(float)*np);
		for(ix=0;ix<nx;ix++)
			for(iz=1;iz<nz;iz++)
			{
				ip=ix*nz+iz;
				mn[ip]=(vp_old[ip]-vp_old[ip-1])/(vp_old[ip]+vp_old[ip-1]);
			}

		fp=fopen("./output/true_vertical_ref.bin","wb");
		fwrite(&mn[0],sizeof(float),np,fp);
		fclose(fp);

		fp=fopen(vp0_file,"rb");
		fread(&vp[0],sizeof(float),np,fp);
		fclose(fp);

		fp=fopen("./output/ini_vp.bin","wb");
		fwrite(&vp[0],sizeof(float),np,fp);
		fclose(fp);

		for(ip=0;ip<np;ip++)
			mn[ip]=(vp_old[ip]-vp[ip])/vp_old[ip];

		fp=fopen("./output/acc_ref.bin","wb");
		fwrite(&mn[0],sizeof(float),np,fp);
		fclose(fp);

		/*fp=fopen(rho_file,"rb");
		  fread(&rho[0],sizeof(float),np,fp);
		  fclose(fp);*/
		for(ip=0;ip<np;ip++)
			rho[ip]=1000.0;

		///////////////////////////////////////////////////////
		///////////////////////////////////////////////////////
		fp=fopen("./output/acc_rho.bin","wb");
		fwrite(&rho[0],sizeof(float),np,fp);
		fclose(fp);

		printf("The true model is done\n"); 

		///////////////////////////////////////////////////////
		fp=fopen(migreal_file,"rb");
		fread(&migreal[0],sizeof(float),np,fp);
		fclose(fp);

		fp=fopen(migpsf_file,"rb");
		fread(&migpsf[0],sizeof(float),np,fp);
		fclose(fp);
	}//end myid

	MPI_Barrier(comm);
	MPI_Bcast(vp,np,MPI_FLOAT,0,comm);
	MPI_Bcast(vp_old,np,MPI_FLOAT,0,comm);
	MPI_Bcast(rho,np,MPI_FLOAT,0,comm);
	MPI_Bcast(mn,np,MPI_FLOAT,0,comm);

	MPI_Bcast(migreal,np,MPI_FLOAT,0,comm);
	MPI_Bcast(migpsf,np,MPI_FLOAT,0,comm);

	/////////////////////////////////////////////////////
	vp_max=0.0;
	for(ip=0;ip<np;ip++)
	{     
		if(vp[ip]>=vp_max)
		{
			vp_max=vp[ip];
		}
	}
	if(myid==0)
	{
		printf("vp_max = %f\n",vp_max); 
	}

	/*=======================================================
	  Phase shift for PSF or not...
	  ========================================================*/

	///////////////////////////////////////////////////////

	/*if(psf_shift_flag==1)
	{
		psf_phase_shift(nx,nz,migpsf);
	}
	*/

	//==========================================================//
	///////////////     Expand the model    /////////////////////
	//==========================================================//
	
	for(ix=0;ix<nx;ix++)
	{
		for(iz=0;iz<nz;iz++)
		{
			ip=ix*nzp+iz;
			ipp=ix*nz+iz;
			
			vpf[ip]=vp[ipp];
			migrealf[ip]=migreal[ipp];
			migpsff[ip]=migpsf[ipp];
		}
	}
	//expand
	for(ix=nx;ix<nxp;ix++)
	{
		for(iz=0;iz<nz;iz++)
		{
			ip=ix*nzp+iz;
			ipp=(nx-1)*nzp+iz;

			vpf[ip]=vpf[ipp];0.0;//
			migrealf[ip]=migrealf[ipp];
			migpsff[ip]=migpsff[ipp];
		}
	}
	for(iz=nz;iz<nzp;iz++)
	{
		for(ix=0;ix<nxp;ix++)
		{
			ip=ix*nzp+iz;
			ipp=ix*nzp+nz-1;

			vpf[ip]=vpf[ipp];
			migrealf[ip]=migrealf[ipp];
			migpsff[ip]=migpsff[ipp];
		}
	}
	if(myid==0)
		printf("The expaned nxp == %d, nzp == %d\n",nxp,nzp);
	//
	//////////////////////////////////////////
	if(myid==0)
	{
		fp=fopen("./output/migration_psf.bin","wb");
		fwrite(&migpsf[0],sizeof(float),np,fp);
		fclose(fp);
	}

	//==========================================================//
	///////////////     Define the window    /////////////////////
	//==========================================================//
	
	memset(coefr,0.0,sizeof(float)*outwinx*outwinz);
	memset(coefp,0.0,sizeof(float)*psfwinx*psfwinz);

	float coefx,coefz,coefmax;
	//////////////////////////////////////////////////////////////
	coefmax=0.0;
	for(ix=0;ix<outwinx;ix++)
	{
		for(iz=0;iz<outwinz;iz++)
		{
			ip=ix*outwinz+iz;

			if(ix<=snapwinx-1)
				coefx=0.5000-0.5*cos(PI*ix/(snapwinx-1));
			else if(ix>=outwinx-snapwinx)
				coefx=0.5000+0.5*cos(PI*(ix-outwinx+snapwinx)/(snapwinx-1));
			else
				coefx=1.0;

			if(iz<=snapwinz-1)
				coefz=0.5000-0.5*cos(PI*iz/(snapwinz-1));
			else if(iz>=outwinz-snapwinz)
				coefz=0.5000+0.5*cos(PI*(iz-outwinz+snapwinz)/(snapwinz-1));
			else
				coefz=1.0;

			coefr[ip]=coefx*coefz;//1.0;//
		}
	}
	/////////coefficient for psf window//////////////
	float hf,sigmaf;
	hf=(halfpsfwx+halfpsfwz)/2.0;
	sigmaf=1.0*hf;   // big for more psf information
	for(ix=0;ix<psfwinx;ix++)
	{
		for(iz=0;iz<psfwinz;iz++)
		{
			ip=ix*psfwinz+iz;

			coefx=(halfpsfwx-ix)*(halfpsfwx-ix);
			coefz=(halfpsfwz-iz)*(halfpsfwz-iz);
			coefp[ip]=1.0;
		}
	}
	/////////////////////////////////////////////////

	if(myid==0){
		fp=fopen("./output/coefr.bin","wb");
		fwrite(&coefr[0],sizeof(float),outwinx*outwinz,fp);
		fclose(fp);
		fp=fopen("./output/psfcoef.bin","wb");
		fwrite(&coefp[0],sizeof(float),psfwinx*psfwinz,fp);
		fclose(fp);
	}

    /// filter the PSF
	for(ixf=0;ixf<nxwpsf;ixf++)
	{
		for(izf=0;izf<nzwpsf;izf++)
		{
			ix=halfpsfwx+ixf*psfwinx;
			iz=halfpsfwz+izf*psfwinz;

			for(ixx=0;ixx<psfwinx;ixx++)
			{
				for(izz=0;izz<psfwinz;izz++)
				{
					ip=ixx*psfwinz+izz;

					/////////////////////////////////////////////////
					//filtered PSFs
					ipp=(ix+ixx-halfpsfwx)*nzp+(iz+izz-halfpsfwz);

					if((ix+ixx-halfpsfwx)<nxp&&(iz+izz-halfpsfwz)<nzp)
					{
						migpsffl[ipp]=migpsff[ipp];///max_coefp;
						migcoefp[ipp]=coefp[ip];
					}
				}
			}//end ixx
		}
	}//end ixf
	if(myid==0)
	{

		fp=fopen("./output/migration_psf_filt.bin","wb");
		for(ix=0;ix<nx;ix++)
			for(iz=0;iz<nz;iz++)
			{
				ip=ix*nzp+iz;
				fwrite(&migpsffl[ip],sizeof(float),1,fp);
			}
		fclose(fp);

		fp=fopen("./output/migration_coefp.bin","wb");
		for(ix=0;ix<nx;ix++)
			for(iz=0;iz<nz;iz++)
			{
				ip=ix*nzp+iz;
				fwrite(&migcoefp[ip],sizeof(float),1,fp);
			}
		fclose(fp);
	}

	//==========================================================//
	/////////     Deconvlution in Wavenumber Domain    ///////////
	//==========================================================//
	
	if(myid==0)
	{
		printf("=========================================\n");
		printf("      =============================\n");
		printf("      Calculate the Deconvolution \n      with Point Spread Functions !\n");
		printf("  -------------------------------------\n");
	}
	//////////////////////////////////////////////////////////////
	memset(psf_inv,0.0,sizeof(float)*npf);

	//////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////
	fftw_plan planff,planbf;
	fftw_plan planft,planb_dec;
	fftw_plan planfm;

	double *snapshot=(double *)malloc(sizeof(double)*psfwinx*psfwinz);// SVD Snap

	fftw_complex *snapm   =(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*fullwx*fullwz);
	fftw_complex *decon   =(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*fullwx*fullwz);

	fftw_complex *snapsf  =(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*fullwx*fullwz);

	fftw_complex *snapsf1 =(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*fullwx*fullwz);
	fftw_complex *snapsf2 =(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*fullwx*fullwz);

	planff=fftw_plan_dft_2d(fullwz,fullwx,snapsf,snapsf, FFTW_FORWARD, FFTW_MEASURE);
	planfm=fftw_plan_dft_2d(fullwz,fullwx,snapm,  snapm, FFTW_FORWARD, FFTW_MEASURE);
	planbf=fftw_plan_dft_2d(fullwz,fullwx,snapsf,snapsf,FFTW_BACKWARD,FFTW_MEASURE);
	planb_dec=fftw_plan_dft_2d(fullwz,fullwx,decon, decon, FFTW_BACKWARD,FFTW_MEASURE);

	float tmpf,fft_max,pfft_max;
	float tmpfloat;
	int downshift =-halfpsfwx;
	int rightshift=-halfpsfwz;
	///////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////
	int nsid,modsr,eachsid,offsets;

	nsid=totalnum/numprocs;
	modsr=totalnum%numprocs;
	if(myid<modsr)
	{
		eachsid=nsid+1;

		offsets=myid*(nsid+1);
	}
	else
	{
		eachsid=nsid;
		offsets=modsr*(nsid+1)+(myid-modsr)*nsid;
	}
	////////////////////////////////////////////////////////////////////
	if(myid==0)
	{fp=fopen("./output/start.flag","w");
		fclose(fp);}
	////////////////////////////////////////////////////////////////////
	
	int averagenum=0;

	int istep,stepindex;

	////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////
	if(myid==0)
	{
		printf("  -------------------------------------------\n");
		printf("Caculate the Deconvolution with the Wavenumber-domain PSFs !\n");
	}

	float P[nzp];
	Preprocess(nzp, nxp, P);

	memset(mig_dec,0.0,sizeof(float)*npf);
	memset(mig_coef,0.0,sizeof(float)*npf);

	for(istep=0;istep<eachsid;istep++)
	{
		stepindex=offsets+istep;

		ix=stepwinx*(stepindex/stepnumz);
		iz=stepwinz*(stepindex%stepnumz);

		memset(snapm,0.0,sizeof(fftw_complex)*fullwx*fullwz);
		//////////////// ------------ ///////////////
		memset(snapsf,0.0,sizeof(fftw_complex)*fullwx*fullwz);

        float pp_max=0.0;
//#pragma omp parallel for private(ixx,izz,ip,iix,iiz,ipp,ipc)
		for(ixx=0;ixx<filterwinx;ixx++)
		{
			for(izz=0;izz<filterwinz;izz++)
			{
				iix=ix-halfltwx+ixx;
				iiz=iz-halfltwz+izz;

				ip=(midpx-halfltwx+ixx)*fullwz+midpz-halfltwz+izz;
				ipp=iix*nzp+iiz;

				if(iix>=0&&iix<nxp&&iiz>=0&&iiz<nzp)
                {
					snapm[ip][0]=migrealf[ipp];//
                    pp_max+=fabs(snapm[ip][0]);
                }
			}//end izz
		}//end ixx
        if(pp_max<=1.0e-15)
            pp_max=1.0e-10;
        for(ip=0;ip<fullwx*fullwz;ip++)
            snapm[ip][0]/=pp_max;//(pp_max/(ii+psfwinx+psfwinz));
		fftw_execute(planfm);

		////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////
		//////////////// ------------ ///////////////
		memset(snapsf  ,0.0,sizeof(fftw_complex)*fullwx*fullwz);
		memset(snapsf1 ,0.0,sizeof(fftw_complex)*fullwx*fullwz);
		memset(snapsf2 ,0.0,sizeof(fftw_complex)*fullwx*fullwz);

        ixf=(ix-halfpsfwx)/psfwinx;
        izf=(iz-halfpsfwz)/psfwinz;
        if(ixf<=2)
            ixf=3;
        if(ixf>=nxwpsf-3)
            ixf=nxwpsf-4;
        if(izf<=2)
            izf=3;
        if(izf>=nzwpsf-3)
            izf=nzwpsf-4;

        ixp=halfpsfwx+ixf*psfwinx;
        izp=halfpsfwz+izf*psfwinz;

        for(ixx=0;ixx<psfwinx;ixx++)
        {
            for(izz=0;izz<psfwinz;izz++)
            {
                //ip=ixx*psfwinz+izz;
                //ip=(midpx-halffwx+ixx)*fullwz+midpz-halffwz+izz;
                ip =ixx*fullwz+izz;
                ipc=ixx*psfwinz+izz;

                ///////////interpolate along Z-direction///////////////
                ipp=(ixp+ixx-halfpsfwx)*nzp+(izp+izz-halfpsfwz);

                snapsf[ip][0]=migpsff[ipp];///maxcoefp;
            }
        }

		////////////////////////////////////////////////

        float psf_max=0.0;
        for(ip=0;ip<fullwx*fullwz;ip++)
            psf_max+=fabs(snapsf[ip][0]);///(fullwx*fullwz);
        for(ip=0;ip<fullwx*fullwz;ip++)
            snapsf[ip][0]/=psf_max;//(psf_max/sqrt(fabs(iz+halfpsfwz)+1.0));//
		fftw_execute(planff);

		fft_max=0.0;
		for(ixx=0;ixx<fullwx;ixx++)
			for(izz=0;izz<fullwz;izz++)
			{
				ip=ixx*fullwz+izz;
				tmpfloat=(snapsf[ip][0]*snapsf[ip][0]+snapsf[ip][1]*snapsf[ip][1]);//
				if(fft_max<tmpfloat)
					fft_max=tmpfloat;
			}

		//fft_max*=epsilon*afft_max[(stepindex/stepnumz)*stepnumz+(stepindex%stepnumz)]*1e+2*(nzp-iz);
		fft_max*=epsilon;///
		if(fft_max==0.0)
			fft_max=1.0e-15;
		////////////////////////////////////////////////////
		////////////////////////////////////////////////////

//#pragma omp parallel for private(ixx,izz,ip,tmpfloat)
		for(ixx=0;ixx<fullwx;ixx++)
			for(izz=0;izz<fullwz;izz++)
			{
				ip=ixx*fullwz+izz;

				tmpfloat=(snapsf[ip][0]*snapsf[ip][0]+snapsf[ip][1]*snapsf[ip][1])+fft_max;//*powf(P[iz],6.0);

				decon[ip][0]=( snapm[ip][0]*snapsf[ip][0]+snapm[ip][1]*snapsf[ip][1])/tmpfloat;
				decon[ip][1]=(-snapm[ip][0]*snapsf[ip][1]+snapm[ip][1]*snapsf[ip][0])/tmpfloat;
			}
		fftw_execute(planb_dec);
		/////////////////////////////////////////////
//#pragma omp parallel for private(ixx,izz,ip,iix,iiz,ipp,ipc)
		for(ixx=0;ixx<outwinx;ixx++)
			for(izz=0;izz<outwinz;izz++)
			{
                ip=(halfltwx-halfoutwx+ixx)*fullwz+halfltwz-halfoutwz+izz;
				ipc=ixx*outwinz+izz;

				iix=ix-halfltwx+halfltwx-halfoutwx+ixx;
				iiz=iz-halfltwz+halfltwz-halfoutwz+izz;
				//iix=ix+halfltwx-halfoutwx+ixx;
				//iiz=iz+halfltwz-halfoutwz+izz;
				ipp=iix*nzp+iiz;

				if(iix>=0&&iix<nxp&&iiz>=0&&iiz<nzp)
				{
					mig_dec[ipp]+=decon[ip][0]*coefr[ipc];
					mig_coef[ipp]+=coefr[ipc];
				}
			}//end izz
	}//end istep
	//
	////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////
	//
	////////////////////////////////////////////////////////////////////
	MPI_Barrier(comm);
	MPI_Allreduce(MPI_IN_PLACE,mig_dec, npf,MPI_FLOAT,MPI_SUM,comm);
	MPI_Allreduce(MPI_IN_PLACE,mig_coef,npf,MPI_FLOAT,MPI_SUM,comm);

	if(myid==0)
	{
		printf("\n");
		fp=fopen("./output/end.flag","w");
		fclose(fp);
	}
	/////// Preprocess for True image ////////

	if(myid==0)
	{
		fp=fopen("./output/migration_pp.bin","wb");
		for(ix=0;ix<nx;ix++)
			for(iz=0;iz<nz;iz++)
			{
				ip=ix*nzp+iz;
				fwrite(&migrealf[ip],sizeof(float),1,fp);
			}
		fclose(fp);

		fp=fopen("./output/migration_coef.bin","wb");
		for(ix=0;ix<nx;ix++)
			for(iz=0;iz<nz;iz++)
			{
				ip=ix*nzp+iz;
				fwrite(&mig_coef[ip],sizeof(float),1,fp);
			}
		fclose(fp);

		fp=fopen("./output/migration_coef.txt","w");
		for(iz=0;iz<nz;iz++)
		{
			for(ix=0;ix<nx;ix++)
			{
				ip=ix*nzp+iz;
				fprintf(fp,"%f  ",mig_coef[ip]);
			}
			fprintf(fp,"\r\t\n");
		}
		fclose(fp);

		fp=fopen(decon_file,"wb");
		for(ix=0;ix<nx;ix++)
			for(iz=0;iz<nz;iz++)
			{
				ip=ix*nzp+iz;
				fwrite(&mig_dec[ip],sizeof(float),1,fp);
			}
		fclose(fp);
	}

	if(myid==0)
	{
		printf("            Deconvolution END!     \n");
		printf("      =============================\n");
		printf("=========================================\n");
	}

	MPI_Barrier(comm);
	//////////////// TEST /////////////////
	/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	  !	        CONVOLUTIOS ENDS...                        !
	  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

	//free the memory of P velocity
	free(vp);
	free(vp_old);
	//free the memory of Density
	free(rho); 
	//free the memory of lamda+2Mu

	free(migreal); 
	free(migpsf); 

	free(migrealf); 
	free(migpsff); 

	free(migcoefp); 

	free(mig_dec);
	free(psf_inv);

	free(coefr);
	free(coefp);
	free(afft_max);

	free(vpf);
	fftw_free(snapm);
	fftw_free(decon);

	fftw_free(snapsf);

	fftw_free(snapsf1);
	fftw_free(snapsf2);

	fftw_destroy_plan(planff);
	fftw_destroy_plan(planfm);
	fftw_destroy_plan(planbf);
	fftw_destroy_plan(planb_dec);

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
	float alpha_max=PI*f0;

	float Vpmax;


	float *alpha_x,*alpha_x_half;
	float *alpha_z,*alpha_z_half;

	float x_start,x_end,delta_x;
	float z_start,z_end,delta_z;
	float x_current,z_current;

	Vpmax=5500;

	thickness_of_pml=pml*dx;

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

	z_start=pml*dx;
	z_end=(ntz-pml-1)*dx;

	// Integer points
	for(iz=0;iz<ntz;iz++)
	{ 
		z_current=iz*dx;

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
		z_current=(iz+0.5)*dx;

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

void get_acc_model(float *vp, float *rho, int ntp, int ntx, int ntz, int pml, char *vp_file, char *rho_file)
{
	int ip,ipp,iz,ix;
	// THE MODEL    
	FILE *fp;

	fp=fopen(vp_file,"rb");
	for(ix=pml;ix<ntx-pml;ix++)
	{
		for(iz=pml;iz<ntz-pml;iz++)
		{
			ip=iz*ntx+ix;
			fread(&vp[ip],sizeof(float),1,fp);           
		}
	}
	fclose(fp);

	for(iz=0;iz<=pml-1;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ix;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}
	}

	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}

	}

	for(iz=ntz-pml;iz<ntz;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ix;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}
	}
	///////////
	/*	fp=fopen(rho_file,"rb");
		for(ix=pml;ix<ntx-pml;ix++)
		{
		for(iz=pml;iz<ntz-pml;iz++)
		{
		ip=iz*ntx+ix;
		fread(&rho[ip],sizeof(float),1,fp);

		rho[ip]=rho[ip];
		}
		}
		fclose(fp);

		for(iz=0;iz<=pml-1;iz++)
		{

		for(ix=0;ix<=pml-1;ix++)
		{
		ip=iz*ntx+ix;
		ipp=pml*ntx+pml;

		rho[ip]=rho[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
		ip=iz*ntx+ix;
		ipp=pml*ntx+ix;

		rho[ip]=rho[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
		ip=iz*ntx+ix;
		ipp=pml*ntx+ntx-pml-1;

		rho[ip]=rho[ipp];
		}
		}

		for(iz=pml;iz<=ntz-pml-1;iz++)
		{
		for(ix=0;ix<=pml-1;ix++)
		{
		ip=iz*ntx+ix;
		ipp=iz*ntx+pml;

		rho[ip]=rho[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
		ip=iz*ntx+ix;
		ipp=iz*ntx+ntx-pml-1;

		rho[ip]=rho[ipp];
		}

		}

		for(iz=ntz-pml;iz<ntz;iz++)
		{

		for(ix=0;ix<=pml-1;ix++)
		{
		ip=iz*ntx+ix;
		ipp=(ntz-pml-1)*ntx+pml;

		rho[ip]=rho[ipp];
		}

	for(ix=pml;ix<=ntx-pml-1;ix++)
	{
		ip=iz*ntx+ix;
		ipp=(ntz-pml-1)*ntx+ix;

		rho[ip]=rho[ipp];
	}

	for(ix=ntx-pml;ix<ntx;ix++)
	{
		ip=iz*ntx+ix;
		ipp=(ntz-pml-1)*ntx+ntx-pml-1;

		rho[ip]=rho[ipp];
	}
}
*/
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
	int   it;
	float temp,max=0.0;

	FILE *fp;

	if(flag==3)
	{	
		for(it=0;it<itmax;it++)
		{
			temp=1.5*PI*f0*(it*dt-t0);
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
			temp=PI*f0*(it*dt-t0);
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
			temp=PI*f0*(it*dt-t0);
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


/*=======================================================================

  subroutine preprocess(nz,nx,dx,dz,P)

  !=======================================================================*/
// in this program Precondition P is computed

void Preprocess(int nz, int nx, float *P)
{
	float dx=1.0,dz=1.0;
	int iz,iz_depth_one,iz_depth_two;
	float z,delta1,a,c,temp,z1,z2;

	float zmax=1.0e+3;
	float cf=sqrt(log(zmax))/nz;

	a=3.0;
	c=1.0;
	iz_depth_one=5;
	iz_depth_two=27;

	delta1=(iz_depth_two-iz_depth_one)*dx;
	z1=(iz_depth_one-1)*dz;
	z2=(iz_depth_two-1)*dz;

	for(iz=0;iz<nz;iz++)
	{ 
		z=iz*dz;
		if(z<=z1)
		{
			P[iz]=1.0;
		}
		else
		{
			temp=(z-z1-delta1)*cf;
			temp=temp*temp;
			P[iz]=c*exp(temp);//0.0;//
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

void input_parameters(int *nx,int *nz,int *psfwinx,int *psfwinz,int *filterwinx,int *filterwinz,int *snapwinx,int *snapwinz,float *epsilon,char *vp_file,char *vp0_file,char *rho_file,char *migreal_file,char *migpsf_file,char *decon_file,char *ipsf_file, int *shift_flag,float *reg_eps
		)
{
	char strtmp[256];
	FILE *fp=fopen("parameter.txt","r");
	if(fp==0)
	{
		printf("Cannot open the parameters file!\n");
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
	fscanf(fp,"%d",psfwinx);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",psfwinz);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",filterwinx);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",filterwinz);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",snapwinx);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",snapwinz);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%f",epsilon);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%s",vp_file);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%s",vp0_file);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%s",rho_file);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%s",migreal_file);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%s",migpsf_file);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%s",decon_file);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%s",ipsf_file);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",shift_flag);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%f",reg_eps);
	fscanf(fp,"\n");

	return;
}

/*void psf_phase_shift(int nx, int nz, float *migpsf)
{
	int ix,iz,ip,ipp;
	int np=nx*nz;

	fftw_complex *migpsf_f=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*nz);
	fftw_complex *migpsf_p=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*nz);

	fftw_plan planfp,planbp;

	planfp=fftw_plan_dft_1d(nz,migpsf_f, migpsf_f, FFTW_FORWARD, FFTW_MEASURE);
	planbp=fftw_plan_dft_1d(nz,migpsf_p, migpsf_p, FFTW_BACKWARD,FFTW_MEASURE);

	for(ix=0;ix<nx;ix++)
	{
		for(iz=0;iz<nz;iz++)
		{
			migpsf_f[iz][0]=migpsf[ix*nz+iz];
			migpsf_f[iz][1]=0.0;
			migpsf_p[iz][0]=0.0;
			migpsf_p[iz][1]=0.0;
		}
		fftw_execute(planfp);

		for(iz=0;iz<nz;iz++)
		{
			if(iz<round(nz/2)){
				migpsf_p[iz][0]= migpsf_f[iz][1];
				migpsf_p[iz][1]=-migpsf_f[iz][0];
			}
			else{
				migpsf_p[iz][0]=-migpsf_f[iz][1];
				migpsf_p[iz][1]= migpsf_f[iz][0];
			}
		}
		fftw_execute(planbp);

		for(iz=0;iz<nz;iz++)
			migpsf[ix*nz+iz]=migpsf_p[iz][0]/nz;
	}//end ix

	fftw_free(migpsf_f);
	fftw_free(migpsf_p);

	fftw_destroy_plan(planfp);
	fftw_destroy_plan(planbp);
}*/

void psf_phase_shift(int nx, int nz, float *migpsf)
{
	int ix,iz,ip,ipp;
	int np=nx*nz;
	int zft,nfft;
	zft=(int)ceil(log(1.0*nz)/log(2.0));
	nfft=(int)pow(2.0,zft);//nz;//
	int halfnz=nz/2;
	int halfft=nfft/2;

	fftw_complex *migpsf_f=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*nfft);
	fftw_complex *migpsf_p=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*nfft);

	fftw_plan planfp,planbp;

	planfp=fftw_plan_dft_1d(nfft,migpsf_f, migpsf_f, FFTW_FORWARD, FFTW_MEASURE);
	planbp=fftw_plan_dft_1d(nfft,migpsf_p, migpsf_p, FFTW_BACKWARD,FFTW_MEASURE);

	for(ix=0;ix<nx;ix++)
	{
		for(iz=0;iz<nfft;iz++)
		{
			migpsf_f[iz][0]=0.0;//migpsf[ix*nz+iz];
			migpsf_f[iz][1]=0.0;
			migpsf_p[iz][0]=0.0;
			migpsf_p[iz][1]=0.0;
		}
		for(iz=0;iz<nz;iz++)
			migpsf_f[halfft-halfnz+iz][0]=migpsf[ix*nz+iz];
			//migpsf_f[iz][0]=migpsf[ix*nz+iz];

		fftw_execute(planfp);

		for(iz=0;iz<nfft;iz++)
		{
			if(iz<round(nfft/2)){
				migpsf_p[iz][0]= migpsf_f[iz][1];
				migpsf_p[iz][1]=-migpsf_f[iz][0];
			}
			else{
				migpsf_p[iz][0]=-migpsf_f[iz][1];
				migpsf_p[iz][1]=-migpsf_f[iz][0];
			}
		}
		fftw_execute(planbp);

		for(iz=0;iz<nz;iz++)
			migpsf[ix*nz+iz]=migpsf_p[halfft-halfnz+iz][0]/nfft;
			//migpsf[ix*nz+iz]=migpsf_p[iz][0]/nfft;
			//migpsf[ix*nz+iz]=migpsf_p[iz][0]/nz;
	}//end ix

	fftw_free(migpsf_f);
	fftw_free(migpsf_p);

	fftw_destroy_plan(planfp);
	fftw_destroy_plan(planbp);
}

void circshift(fftw_complex *snapshift, int filterwinz, int filterwinx, fftw_complex *snapf, int downshift, int rightshift)
{
	int newshiftz,newshiftx;
	newshiftz=(( downshift%filterwinz)+filterwinz)%filterwinz;
	newshiftx=((rightshift%filterwinx)+filterwinx)%filterwinz;
	int iz,ix;
	int izz,ixx;
	int ip,ipp;

	for(ix=0;ix<filterwinx;ix++)
	{
		ixx=(ix+newshiftx)%filterwinx;
		for(iz=0;iz<filterwinz;iz++)
		{
			izz=(iz+newshiftz)%filterwinz;

			ip=ix*filterwinz+iz;
			ipp=ixx*filterwinz+izz;

			snapshift[ipp][0]=snapf[ip][0];
		}
	}
}


