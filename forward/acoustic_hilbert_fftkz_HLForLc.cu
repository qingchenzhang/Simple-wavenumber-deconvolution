#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "headrtm.h"
#include "cufft.h"

//#define BATCH 834

#define reference_window 1 //0--without 1--with

struct Multistream
{
	cudaStream_t stream,stream_back;
};

__global__ void hilbert_transform2d_kernel(cufftComplex *p, cufftComplex *p_hilbert, 
		int ntz, int ntx)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ic;
	int ip=ix*ntz+iz;
	int halfz=ceilf(ntz/2);

	if(iz>0&&iz<=halfz&&ix>=0&&ix<=ntx)
	{
		ip=ix*ntz+iz;

		p_hilbert[ip].x= p[ip].y;
		p_hilbert[ip].y=-p[ip].x;
	}  
	if(iz>halfz&&iz<ntz&&ix>=0&&ix<=ntx)
	{

		p_hilbert[ip].x=-p[ip].y;
		p_hilbert[ip].y= p[ip].x;
	}  

	__syncthreads();
}

__global__ void fft_kz_sep_kernel(cufftComplex *p, cufftComplex *pu, cufftComplex *pd, 
		float *coef,
		int ntz, int ntx)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ic;
	int ip=ix*ntz+iz;
	int halfz=ceilf(ntz/2.0);

	if(iz>=0&&iz<=halfz&&ix>=0&&ix<=ntx)
	{
		ip=ix*ntz+iz;

		pu[ip].x= p[ip].x*coef[iz];
		pu[ip].y= p[ip].y*coef[iz];

		pd[ip].x= 0.0;
		pd[ip].y= 0.0;
	}  
	if(iz>halfz&&iz<ntz&&ix>=0&&ix<=ntx)
	{
		pu[ip].x= 0.0;
		pu[ip].y= 0.0;

		pd[ip].x= p[ip].x*coef[iz];
		pd[ip].y= p[ip].y*coef[iz];
	}  

	__syncthreads();
}

__global__ void output_seismogram_kernel(
		cufftComplex *spu, cufftComplex *spd,
		float *seismogram_up, float *seismogram_down, int r_ix, int *r_iz, int r_n,
		int pml, int Lc, float *rc, 
		int ntx, int ntz, int nfftz, int it, int itmax
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int pmlc=pml+Lc;

	int nx=ntx-2*pmlc;
	int nz=ntz-2*pmlc;

	float dvx_dx,dvz_dz;
	int ic,ii;
	int ip=ix*ntz+iz;
	int ipp=ix*nfftz+iz;

	// Seismogram...   
	//if(ix==r_ix)
	if(ix==pmlc)
		for(ii=0;ii<r_n/2;ii++)
		{
			if(iz==r_iz[ii])
			{
				seismogram_up[ii*itmax+it]  =spu[ix*nfftz+iz].x;
				seismogram_down[ii*itmax+it]=spd[ix*nfftz+iz].x;
			}
		}

	if(ix==ntx-pmlc-1)
		for(ii=0;ii<r_n/2;ii++)
		{
			if(iz==r_iz[ii])
			{
				seismogram_up[(ii+r_n/2)*itmax+it]  =spu[ix*nfftz+iz].x;
				seismogram_down[(ii+r_n/2)*itmax+it]=spd[ix*nfftz+iz].x;
			}
		}
}

__global__ void wavefield_seperation_kernel(cufftComplex *p, cufftComplex *hp, cufftComplex *htp, cufftComplex *hhtp, cufftComplex *pu, cufftComplex *pd, 
		int ntz, int ntx)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ic;
	int ip=ix*ntz+iz;

	if(iz>=0&&iz<ntz&&ix>=0&&ix<=ntx)
	{
		pu[ip].x=0.5*(p[ip].x-hhtp[ip].x/ntz);
		pu[ip].y=0.5*(hp[ip].x/ntz+htp[ip].x);

		pd[ip].x=0.5*(p[ip].x+hhtp[ip].x/ntz);
		pd[ip].y=0.5*(hp[ip].x/ntz-htp[ip].x);
	}  

	__syncthreads();
}


__global__ void fdtd_cpml_2d_GPU_kernel_vx(
		float *rho, int *mx, int itmax,
		float *a_x_half, float *a_z, 
		float *b_x_half, float *b_z, 
		float *vx, cufftComplex *p,
		float *phi_p_x,
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		int it, int pml, int Lc, float *rc
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ic;
	int ip=ix*ntz+iz;

	float one_over_dx=1.0/dx;

	float dp_dx,one_over_rho_half_x;

	if(iz>=Lc&&iz<=ntz-Lc&&ix>=Lc-1&&ix<=ntx-Lc-1)
	{
		dp_dx=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dp_dx+=rc[ic]*(p[ip+(ic+1)*ntz].x-p[ip-ic*ntz].x)*one_over_dx;
		}

		phi_p_x[ip]=b_x_half[ix]*phi_p_x[ip]+a_x_half[ix]*dp_dx;

		one_over_rho_half_x=1/(0.5*(rho[ip]+rho[ip+1*ntz]));

		vx[ip]=dt*one_over_rho_half_x*(dp_dx+phi_p_x[ip])+vx[ip];
	}   

	__syncthreads();

}


__global__ void fdtd_cpml_2d_GPU_kernel_vz(
		float *rho, int *mx, int itmax,
		float *a_x, float *a_z_half,
		float *b_x, float *b_z_half,
		float *vz, cufftComplex *p, 
		float *phi_p_z,
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		int it, int pml, int Lc, float *rc
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dz=1.0/dz;

	float dp_dz,one_over_rho_half_z;

	int ic;
	int ip=ix*ntz+iz;

	if(iz>=Lc-1&&iz<=ntz-Lc-1&&ix>=Lc&&ix<=ntx-Lc)
	{
		dp_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dp_dz+=rc[ic]*(p[ip+(ic+1)].x-p[ip-ic].x)*one_over_dz;
		}

		phi_p_z[ip]=b_z_half[iz]*phi_p_z[ip]+a_z_half[iz]*dp_dz;

		one_over_rho_half_z=1/(0.5*(rho[ip]+rho[ip+1]));

		vz[ip]=dt*one_over_rho_half_z*(dp_dz+phi_p_z[ip])+vz[ip];
	}

	__syncthreads();

}


__global__ void fdtd_cpml_2d_GPU_kernel_p(
		float *rick, float *vp, float *rho, int *mx,
		float *a_x, float *a_z,
		float *b_x, float *b_z,
		float *vx, float *vz, cufftComplex *p, cufftComplex *temp,
		float *phi_vx_x, float *phi_vz_z, 
		int ntp, int ntx, int ntz, int nfftz,
		float *seismogram, int r_ix, int *r_iz, int r_n,
		int pml, int Lc, float *rc, 
		float dx, float dz, float dt, int s_ix, int s_iz, int it,
		int inv_flag, int itmax,
		float *p_borders_up, float *p_borders_bottom,
		float *p_borders_left, float *p_borders_right
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int pmlc=pml+Lc;

	int nx=ntx-2*pmlc;
	int nz=ntz-2*pmlc;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dz;

	float dvx_dx,dvz_dz;
	int ic,ii;
	int ip=ix*ntz+iz;
	int ipp=ix*nfftz+iz;

	if(iz>=Lc&&iz<=ntz-Lc&&ix>=Lc&&ix<=ntx-Lc)
	{
		dvx_dx=0.0;
		dvz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dvx_dx+=rc[ic]*(vx[ip+ic*ntz]-vx[ip-(ic+1)*ntz])*one_over_dx;
			dvz_dz+=rc[ic]*(vz[ip+ic]-vz[ip-(ic+1)])*one_over_dz;
		}

		phi_vx_x[ip]=b_x[ix]*phi_vx_x[ip]+a_x[ix]*dvx_dx;
		phi_vz_z[ip]=b_z[iz]*phi_vz_z[ip]+a_z[iz]*dvz_dz;

		p[ip].x=dt*rho[ip]*vp[ip]*vp[ip]*(dvx_dx+phi_vx_x[ip]+dvz_dz+phi_vz_z[ip])+p[ip].x;

	}

	if(iz==s_iz&&mx[ix]==s_ix)
	{
		p[ip].x+=rick[it];
	}

	if(ix>=0&&ix<ntx&&iz>=0&&iz<ntz)
	{
		temp[ipp].x=p[ip].x;
	}

	// Seismogram...   
	//if(ix==r_ix)
	if(ix==pmlc)
		for(ii=0;ii<r_n/2;ii++)
		{
			if(iz==r_iz[ii])
			{
				seismogram[ii*itmax+it]=p[ip].x;
			}
		}
	if(ix==ntx-pmlc-1)
		for(ii=0;ii<r_n/2;ii++)
		{
			if(iz==r_iz[ii])
			{
				seismogram[(ii+r_n/2)*itmax+it]=p[ip].x;
			}
		}

	// Borders...
	if(inv_flag==1)
	{
		if(ix>=pmlc&&ix<=ntx-pmlc-1&&iz>=pmlc&&iz<=pmlc+Lc-1)
		{
			p_borders_up[(iz-pmlc)*itmax*nx+it*nx+ix-pmlc]=p[ip].x;
		}
		if(ix>=pmlc&&ix<=ntx-pmlc-1&&iz>=ntz-pmlc-Lc&&iz<=ntz-pmlc-1)
		{
			p_borders_bottom[(iz-ntz+pmlc+Lc)*itmax*nx+it*nx+ix-pmlc]=p[ip].x;
		}

		if(iz>=pmlc+Lc&&iz<=ntz-pmlc-Lc-1&&ix>=pmlc&&ix<=pmlc+Lc-1)
		{
			p_borders_left[(ix-pmlc)*itmax*(nz-2*Lc)+it*(nz-2*Lc)+iz-pmlc-Lc]=p[ip].x;
		}
		if(iz>=pmlc+Lc&&iz<=ntz-pmlc-Lc-1&&ix>=ntx-pmlc-Lc&&ix<=ntx-pmlc-1)
		{
			p_borders_right[(ix-ntx+pmlc+Lc)*itmax*(nz-2*Lc)+it*(nz-2*Lc)+iz-pmlc-Lc]=p[ip].x;
		}
	}
	__syncthreads();
}

__global__ void fdtd_cpml_2d_GPU_kernel_htvx(
		float *rho, int *mx, int itmax,
		float *a_x_half, float *a_z, 
		float *b_x_half, float *b_z, 
		float *htvx, cufftComplex *htp,
		float *phi_htp_x,
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		int it, int pml, int Lc, float *rc
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ic;
	int ip=ix*ntz+iz;

	float one_over_dx=1.0/dx;

	float dhtp_dx,one_over_rho_half_x;

	if(iz>=Lc&&iz<=ntz-Lc&&ix>=Lc-1&&ix<=ntx-Lc-1)
	{
		dhtp_dx=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dhtp_dx+=rc[ic]*(htp[ip+(ic+1)*ntz].x-htp[ip-ic*ntz].x)*one_over_dx;
		}

		phi_htp_x[ip]=b_x_half[ix]*phi_htp_x[ip]+a_x_half[ix]*dhtp_dx;

		one_over_rho_half_x=1/(0.5*(rho[ip]+rho[ip+1*ntz]));

		htvx[ip]=dt*one_over_rho_half_x*(dhtp_dx+phi_htp_x[ip])+htvx[ip];
	}   

	__syncthreads();

}


__global__ void fdtd_cpml_2d_GPU_kernel_htvz(
		float *rho, int *mx, int itmax,
		float *a_x, float *a_z_half,
		float *b_x, float *b_z_half,
		float *htvz, cufftComplex *htp, 
		float *phi_htp_z,
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		int it, int pml, int Lc, float *rc
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dz=1.0/dz;

	float dhtp_dz,one_over_rho_half_z;

	int ic;
	int ip=ix*ntz+iz;

	if(iz>=Lc-1&&iz<=ntz-Lc-1&&ix>=Lc&&ix<=ntx-Lc)
	{
		dhtp_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dhtp_dz+=rc[ic]*(htp[ip+(ic+1)].x-htp[ip-ic].x)*one_over_dz;
		}

		phi_htp_z[ip]=b_z_half[iz]*phi_htp_z[ip]+a_z_half[iz]*dhtp_dz;

		one_over_rho_half_z=1/(0.5*(rho[ip]+rho[ip+1]));

		htvz[ip]=dt*one_over_rho_half_z*(dhtp_dz+phi_htp_z[ip])+htvz[ip];
	}

	__syncthreads();

}


__global__ void fdtd_cpml_2d_GPU_kernel_htp(
		float *rick, float *vp, float *rho, int *mx,
		float *a_x, float *a_z,
		float *b_x, float *b_z,
		float *htvx, float *htvz, cufftComplex *htp, cufftComplex *temp,
		float *phi_htvx_x, float *phi_htvz_z, 
		int ntp, int ntx, int ntz, int nfftz,
		float *seismogram, int r_ix, int *r_iz, int r_n,
		int pml, int Lc, float *rc, 
		float dx, float dz, float dt, int s_ix, int s_iz, int it,
		int inv_flag, int itmax,
		float *htp_borders_up, float *htp_borders_bottom,
		float *htp_borders_left, float *htp_borders_right
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int pmlc=pml+Lc;

	int nx=ntx-2*pmlc;
	int nz=ntz-2*pmlc;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dz;

	float dhtvx_dx,dhtvz_dz;
	int ic,ii;
	int ip=ix*ntz+iz;
	int ipp=ix*nfftz+iz;

	if(iz>=Lc&&iz<=ntz-Lc&&ix>=Lc&&ix<=ntx-Lc)
	{
		dhtvx_dx=0.0;
		dhtvz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dhtvx_dx+=rc[ic]*(htvx[ip+ic*ntz]-htvx[ip-(ic+1)*ntz])*one_over_dx;
			dhtvz_dz+=rc[ic]*(htvz[ip+ic]-htvz[ip-(ic+1)])*one_over_dz;
		}

		phi_htvx_x[ip]=b_x[ix]*phi_htvx_x[ip]+a_x[ix]*dhtvx_dx;
		phi_htvz_z[ip]=b_z[iz]*phi_htvz_z[ip]+a_z[iz]*dhtvz_dz;

		htp[ip].x=dt*rho[ip]*vp[ip]*vp[ip]*(dhtvx_dx+phi_htvx_x[ip]+dhtvz_dz+phi_htvz_z[ip])+htp[ip].x;

	}

	if(iz==s_iz&&mx[ix]==s_ix)
	{
		htp[ip].x+=rick[it];
	}

	if(ix>=0&&ix<ntx&&iz>=0&&iz<ntz)
	{
		temp[ipp].y=htp[ip].x;
	}

	// Seismogram...   

	// Borders...
	if(inv_flag==1)
	{
		if(ix>=pmlc&&ix<=ntx-pmlc-1&&iz>=pmlc&&iz<=pmlc+Lc-1)
		{
			htp_borders_up[(iz-pmlc)*itmax*nx+it*nx+ix-pmlc]=htp[ip].x;
		}
		if(ix>=pmlc&&ix<=ntx-pmlc-1&&iz>=ntz-pmlc-Lc&&iz<=ntz-pmlc-1)
		{
			htp_borders_bottom[(iz-ntz+pmlc+Lc)*itmax*nx+it*nx+ix-pmlc]=htp[ip].x;
		}

		if(iz>=pmlc+Lc&&iz<=ntz-pmlc-Lc-1&&ix>=pmlc&&ix<=pmlc+Lc-1)
		{
			htp_borders_left[(ix-pmlc)*itmax*(nz-2*Lc)+it*(nz-2*Lc)+iz-pmlc-Lc]=htp[ip].x;
		}
		if(iz>=pmlc+Lc&&iz<=ntz-pmlc-Lc-1&&ix>=ntx-pmlc-Lc&&ix<=ntx-pmlc-1)
		{
			htp_borders_right[(ix-ntx+pmlc+Lc)*itmax*(nz-2*Lc)+it*(nz-2*Lc)+iz-pmlc-Lc]=htp[ip].x;
		}
	}
	__syncthreads();
}


/*==========================================================

  This subroutine is used for calculating the forward wave 
  field of 2D in time domain.

  1.
  inv_flag==0----Calculate the observed seismograms of 
  Vx and Vz components...
  2.
  inv_flag==1----Calculate the synthetic seismograms of 
  Vx and Vz components and store the 
  borders of Vx and Vz used for constructing 
  the forward wavefields. 
  ===========================================================*/

extern "C"
void fdtd_2d_GPU_forward(int ntx, int ntz, int ntp, int nx, int nz,
		int pml, int Lc, float *rc, float dx, float dz,
		float *rick, float *hrick, int itmax, float dt, int myid,
		int is, struct Source ss[], struct MultiGPU plan[], int GPU_N, int rnmax,
		float *vp, float *rho, int *mx,
		float *k_x, float *k_x_half,
		float *k_z, float *k_z_half,
		float *a_x, float *a_x_half,
		float *a_z, float *a_z_half,
		float *b_x, float *b_x_half,
		float *b_z, float *b_z_half, 
		float *c, int inv_flag)
{
	int i,it,ip;
	float tmpfloat;
	int ix,iz;
	int pmlc=pml+Lc;

	int KZ=(int)ceil(log(1.0*ntz)/log(2.0));
	int nfftz=pow(2.0,KZ);

	size_t size_model=sizeof(float)*ntp;

	FILE *fp;
	char filename[40];

	Multistream plans[GPU_N];

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		cudaStreamCreate(&plans[i].stream);	

		cufftSetStream(plan[i].forward_z, plans[i].stream);
		cufftSetStream(plan[i].backward_z,plans[i].stream);
	}

	///////////////////////////////
	// initialize the fields........................

	// copy the vectors from the host to the device
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		cudaMemcpyAsync(plan[i].d_coef,plan[i].coef,sizeof(float)*nfftz,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_r_iz,ss[is+i].r_iz,sizeof(int)*ss[is+i].r_n,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_rick,rick,sizeof(float)*itmax,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_hrick,hrick,sizeof(float)*itmax,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_rc,rc,sizeof(float)*Lc,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_vp,vp,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_rho,rho,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_mx,mx,sizeof(int)*ntx,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_a_x,a_x,sizeof(float)*ntx,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_a_x_half,a_x_half,sizeof(float)*ntx,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_a_z,a_z,sizeof(float)*ntz,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_a_z_half,a_z_half,sizeof(float)*ntz,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_b_x,b_x,sizeof(float)*ntx,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_b_x_half,b_x_half,sizeof(float)*ntx,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_b_z,b_z,sizeof(float)*ntz,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_b_z_half,b_z_half,sizeof(float)*ntz,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemsetAsync(plan[i].d_vx,0.0,size_model,plans[i].stream);
		cudaMemsetAsync(plan[i].d_vz,0.0,size_model,plans[i].stream);
		cudaMemsetAsync(plan[i].d_p,0.0,sizeof(cufftComplex)*ntp,plans[i].stream);

		cudaMemsetAsync(plan[i].d_htvx,0.0,size_model,plans[i].stream);
		cudaMemsetAsync(plan[i].d_htvz,0.0,size_model,plans[i].stream);
		cudaMemsetAsync(plan[i].d_htp,0.0,sizeof(cufftComplex)*ntp,plans[i].stream);

		cudaMemsetAsync(plan[i].d_phi_vx_x,0.0,size_model,plans[i].stream);
		cudaMemsetAsync(plan[i].d_phi_vz_z,0.0,size_model,plans[i].stream);

		cudaMemsetAsync(plan[i].d_phi_htvx_x,0.0,size_model,plans[i].stream);
		cudaMemsetAsync(plan[i].d_phi_htvz_z,0.0,size_model,plans[i].stream);

		cudaMemsetAsync(plan[i].d_phi_p_x,0.0,size_model,plans[i].stream);
		cudaMemsetAsync(plan[i].d_phi_p_z,0.0,size_model,plans[i].stream);

		cudaMemsetAsync(plan[i].d_phi_htp_x,0.0,size_model,plans[i].stream);
		cudaMemsetAsync(plan[i].d_phi_htp_z,0.0,size_model,plans[i].stream);

		////////////////Seperated Wavefield ////////////////////
		cudaMemsetAsync(plan[i].d_spu,0.0,sizeof(cufftComplex)*ntx*nfftz,plans[i].stream);
		cudaMemsetAsync(plan[i].d_spd,0.0,sizeof(cufftComplex)*ntx*nfftz,plans[i].stream);

		cudaMemsetAsync(plan[i].d_rpu,0.0,sizeof(cufftComplex)*ntx*nfftz,plans[i].stream);
		cudaMemsetAsync(plan[i].d_rpd,0.0,sizeof(cufftComplex)*ntx*nfftz,plans[i].stream);

	}
	/////////////////////////////////
	// =============================================================================

	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 dimGrid((ntx+dimBlock.x-1)/dimBlock.x,(nfftz+dimBlock.y-1)/dimBlock.y);

	//-----------------------------------------------------------------------//
	//=======================================================================//
	//-----------------------------------------------------------------------//
	for(it=0;it<itmax;it++)
	{
		for(i=0;i<GPU_N;i++)
		{
			cudaSetDevice(i);

			fdtd_cpml_2d_GPU_kernel_vx<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rho, plan[i].d_mx, itmax, plan[i].d_a_x_half, plan[i].d_a_z, 
				 plan[i].d_b_x_half, plan[i].d_b_z, 
				 plan[i].d_vx, plan[i].d_p,
				 plan[i].d_phi_p_x, 
				 ntp, ntx, ntz, dx, dz, dt,
				 it, pml, Lc, plan[i].d_rc
				);

			fdtd_cpml_2d_GPU_kernel_vz<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rho, plan[i].d_mx, itmax,
				 plan[i].d_a_x, plan[i].d_a_z_half,
				 plan[i].d_b_x, plan[i].d_b_z_half,
				 plan[i].d_vz, plan[i].d_p, 
				 plan[i].d_phi_p_z,
				 ntp, ntx, ntz, dx, dz, dt,
				 it, pml, Lc, plan[i].d_rc
				);

			cudaMemsetAsync(plan[i].d_temp,0.0,sizeof(cufftComplex)*ntx*nfftz,plans[i].stream);

			fdtd_cpml_2d_GPU_kernel_p<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rick, plan[i].d_vp, plan[i].d_rho, plan[i].d_mx,
				 plan[i].d_a_x, plan[i].d_a_z, plan[i].d_b_x, plan[i].d_b_z,
				 plan[i].d_vx, plan[i].d_vz, plan[i].d_p, plan[i].d_temp,
				 plan[i].d_phi_vx_x, plan[i].d_phi_vz_z,
				 ntp, ntx, ntz, nfftz,
				 plan[i].d_seismogram, ss[is+i].r_ix, plan[i].d_r_iz, ss[is+i].r_n,
				 pml, Lc, plan[i].d_rc, dx, dz, dt,
				 ss[is+i].s_ix, ss[is+i].s_iz, it,
				 inv_flag, itmax,
				 plan[i].d_p_borders_up, plan[i].d_p_borders_bottom,
				 plan[i].d_p_borders_left, plan[i].d_p_borders_right
				);

			////////////////////////////////////////////
			////////  Hilbert transform in time direction
			fdtd_cpml_2d_GPU_kernel_htvx<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rho, plan[i].d_mx, itmax, plan[i].d_a_x_half, plan[i].d_a_z, 
				 plan[i].d_b_x_half, plan[i].d_b_z, 
				 plan[i].d_htvx, plan[i].d_htp,
				 plan[i].d_phi_htp_x, 
				 ntp, ntx, ntz, dx, dz, dt,
				 it, pml, Lc, plan[i].d_rc
				);

			fdtd_cpml_2d_GPU_kernel_htvz<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rho, plan[i].d_mx, itmax,
				 plan[i].d_a_x, plan[i].d_a_z_half,
				 plan[i].d_b_x, plan[i].d_b_z_half,
				 plan[i].d_htvz, plan[i].d_htp, 
				 plan[i].d_phi_htp_z,
				 ntp, ntx, ntz, dx, dz, dt,
				 it, pml, Lc, plan[i].d_rc
				);

			/////////////////////////////////////////////////////

			fdtd_cpml_2d_GPU_kernel_htp<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_hrick, plan[i].d_vp, plan[i].d_rho, plan[i].d_mx,
				 plan[i].d_a_x, plan[i].d_a_z, plan[i].d_b_x, plan[i].d_b_z,
				 plan[i].d_htvx, plan[i].d_htvz, plan[i].d_htp, plan[i].d_temp,
				 plan[i].d_phi_htvx_x, plan[i].d_phi_htvz_z,
				 ntp, ntx, ntz, nfftz,
				 plan[i].d_seismogram, ss[is+i].r_ix, plan[i].d_r_iz, ss[is+i].r_n,
				 pml, Lc, plan[i].d_rc, dx, dz, dt,
				 ss[is+i].s_ix, ss[is+i].s_iz, it,
				 inv_flag, itmax,
				 plan[i].d_htp_borders_up, plan[i].d_htp_borders_bottom,
				 plan[i].d_htp_borders_left, plan[i].d_htp_borders_right
				);
			////////  FFT transform in z-depth direction
			cufftExecC2C(plan[i].forward_z,plan[i].d_temp,plan[i].d_temp,CUFFT_FORWARD);

			fft_kz_sep_kernel<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(plan[i].d_temp, plan[i].d_spu, plan[i].d_spd, plan[i].d_coef, 
				 nfftz, ntx);

			cufftExecC2C(plan[i].backward_z,plan[i].d_spu,plan[i].d_spu,CUFFT_INVERSE); 
			cufftExecC2C(plan[i].backward_z,plan[i].d_spd,plan[i].d_spd,CUFFT_INVERSE); 
			////////////////////////////////////////////
			////////////////////////////////////////////
			output_seismogram_kernel<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_spu, plan[i].d_spd,
				 plan[i].d_seismogram_up, plan[i].d_seismogram_down, ss[is+i].r_ix, plan[i].d_r_iz, ss[is+i].r_n,
				 pml, Lc, plan[i].d_rc, ntx, ntz, nfftz, it, itmax
				);

			/*if(inv_flag==1&&it%100==0&&i==0)
			{
				printf("it==%d\n",it);
				cudaMemcpyAsync(plan[i].p,plan[i].d_p,sizeof(cufftComplex)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
				cudaMemcpyAsync(plan[i].htp,plan[i].d_htp,sizeof(cufftComplex)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
				cudaMemcpyAsync(plan[i].spu,plan[i].d_spu,sizeof(cufftComplex)*ntx*nfftz,cudaMemcpyDeviceToHost,plans[i].stream);
				cudaMemcpyAsync(plan[i].spd,plan[i].d_spd,sizeof(cufftComplex)*ntx*nfftz,cudaMemcpyDeviceToHost,plans[i].stream);
				cudaStreamSynchronize(plans[i].stream);

				sprintf(filename,"./output/%dp%d.bin",it,i);     
				fp=fopen(filename,"wb");
				for(ix=pmlc;ix<ntx-pmlc;ix++)
				{
					for(iz=pmlc;iz<ntz-pmlc;iz++)
					{					
						tmpfloat=plan[i].p[ix*ntz+iz].x;
						fwrite(&tmpfloat,sizeof(float),1,fp);
					}
				}
				fclose(fp);
				/////////////////////////////////////////////////////////
				sprintf(filename,"./output/%dhtp%d.bin",it,i); 
				fp=fopen(filename,"wb");
				for(ix=pmlc;ix<ntx-pmlc;ix++)
				{
					for(iz=pmlc;iz<ntz-pmlc;iz++)
					{					
						tmpfloat=plan[i].htp[ix*ntz+iz].x;
						fwrite(&tmpfloat,sizeof(float),1,fp);
					}
				}
				fclose(fp);
				/////////////////////////////////////////////////////////
				sprintf(filename,"./output/%dspu%d.bin",it,i);     
				fp=fopen(filename,"wb");
				for(ix=pmlc;ix<ntx-pmlc;ix++)
				{
					for(iz=pmlc;iz<ntz-pmlc;iz++)
					{					
						fwrite(&plan[i].spu[ix*nfftz+iz].x,sizeof(float),1,fp);
					}
				}
				fclose(fp);
				/////////////////////////////////////////////////////////
				sprintf(filename,"./output/%dspd%d.bin",it,i);   
				fp=fopen(filename,"wb");
				for(ix=pmlc;ix<ntx-pmlc;ix++)
				{
					for(iz=pmlc;iz<ntz-pmlc;iz++)
					{					
						fwrite(&plan[i].spd[ix*nfftz+iz].x,sizeof(float),1,fp);
					}
				}
				fclose(fp);
				///////////////////////////////////////////////////////////////////
			}
*/
		}//end GPU_N
	}//end it

	for(i=0;i<GPU_N;i++)
	{
		cudaMemcpyAsync(plan[i].seismogram_obs,plan[i].d_seismogram,
				sizeof(float)*ss[is+i].r_n*itmax,cudaMemcpyDeviceToHost,plans[i].stream);

		cudaMemcpyAsync(plan[i].seismogram_up,plan[i].d_seismogram_up,
				sizeof(float)*ss[is+i].r_n*itmax,cudaMemcpyDeviceToHost,plans[i].stream);
		cudaMemcpyAsync(plan[i].seismogram_down,plan[i].d_seismogram_down,
				sizeof(float)*ss[is+i].r_n*itmax,cudaMemcpyDeviceToHost,plans[i].stream);
	}

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}//end GPU

	//free the memory of DEVICE
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		cudaStreamDestroy(plans[i].stream);
	}
}


/////////Preprocessing for the backward residual seismogram//////
extern "C"
void rmsgpu_fre(float *seismogram_rms,int itmax,int nx,int i,float dt)
{ 
	cudaSetDevice(i);

	int ix,it,ip,K,NX;
	//	float alpha_rms=1.0e-25;

	float pi=3.1415926;

	K=(int)ceil(log(1.0*itmax)/log(2.0));
	NX=(int)pow(2.0,K);

	float dw,kw;
	dw=(float)1.0/((NX)*dt);

	int BATCH=nx;
	int NTP=NX*BATCH;

	cufftComplex *vx,*temp,*tempout;		

	cudaMallocHost((void **)&vx, sizeof(cufftComplex)*NX*BATCH);
	cudaMalloc((void **)&temp,sizeof(cufftComplex)*NX*BATCH);
	cudaMalloc((void **)&tempout,sizeof(cufftComplex)*NX*BATCH);

	cufftHandle plan;
	cufftPlan1d(&plan,NX,CUFFT_C2C,BATCH);

	for(it=0;it<NTP;it++)
	{ 
		vx[it].x=0.0;
		vx[it].y=0.0; 
	} 

	for(ix=0;ix<nx;ix++)
	{            
		for(it=0;it<itmax;it++)
		{
			vx[ix*NX+it].x=seismogram_rms[ix*itmax+it];//;
		}
	} 

	cudaMemcpy(temp,vx,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
	cufftExecC2C(plan,temp,tempout,CUFFT_FORWARD);
	cudaMemcpy(vx,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

	for(ix=0;ix<nx;ix++)
	{            
		for(it=0;it<NX;it++)
		{
			ip=ix*NX+it;

			if(it==0)
			{
				kw=1.0;
			}
			else if(it<NX/2)
			{
				kw=2*pi*it*dw;
			}
			else
			{
				kw=2*pi*(NX-1-it)*dw;
			}

			vx[ip].x=vx[ip].x/(kw*kw);
			vx[ip].y=vx[ip].y/(kw*kw);
		}
	} 

	// fft(r_real,r_imag,NFFT,-1);

	cudaMemcpy(temp,vx,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
	cufftExecC2C(plan,temp,tempout,CUFFT_INVERSE);
	cudaMemcpy(vx,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

	for(ix=0;ix<nx;ix++)
	{
		for(it=0;it<itmax;it++)
		{
			seismogram_rms[ix*itmax+it]=vx[ix*NX+it].x;
		}
	} 
	cudaFreeHost(vx);
	cudaFree(temp);
	cudaFree(tempout);

	cufftDestroy(plan);

	return;
}
//*******************************************************//
//*******************Laplacian Filter********************//
extern "C"
void laplacegpu_filter(float *image, int i, int nxx,
		int nzz, float dx, float dz, int pml)
{ 
	cudaSetDevice(i);

	int ix,iz,ip,K,NX,NZ;

	float pi=3.1415926;

	K=(int)ceil(log(1.0*nxx)/log(2.0));
	NX=(int)pow(2.0,K);

	K=(int)ceil(log(1.0*nzz)/log(2.0));
	NZ=(int)pow(2.0,K);

	float dkx,dkz;
	float kx,kz;

	dkx=(float)1.0/((NX)*dx);
	dkz=(float)1.0/((NZ)*dz);

	int NTP=NX*NZ;

	cufftComplex *pp,*temp,*tempout;		

	cudaMallocHost((void **)&pp, sizeof(cufftComplex)*NX*NZ);
	cudaMalloc((void **)&temp,sizeof(cufftComplex)*NX*NZ);
	cudaMalloc((void **)&tempout,sizeof(cufftComplex)*NX*NZ);

	cufftHandle plan;
	cufftPlan2d(&plan,NX,NZ,CUFFT_C2C);

	for(ip=0;ip<NTP;ip++)
	{ 
		pp[ip].x=0.0;
		pp[ip].y=0.0; 
	} 

	for(ix=0;ix<nxx;ix++)
	{            
		for(iz=0;iz<nzz;iz++)
		{
			pp[ix*NZ+iz].x=image[ix*nzz+iz];
		}
	} 

	cudaMemcpy(temp,pp,sizeof(cufftComplex)*NX*NZ,cudaMemcpyHostToDevice);
	cufftExecC2C(plan,temp,tempout,CUFFT_FORWARD);
	cudaMemcpy(pp,tempout,sizeof(cufftComplex)*NX*NZ,cudaMemcpyDeviceToHost);

	for(ix=0;ix<NX;ix++)
	{            
		for(iz=0;iz<NZ;iz++)
		{
			if(ix<NX/2)
			{
				kx=2*pi*ix*dkx;
			}
			if(ix>NX/2)	
			{
				kx=2*pi*(NX-1-ix)*dkx;
			}

			if(iz<NZ/2)
			{
				kz=2*pi*iz*dkz;//2*pi*(NZ/2-1-iz)*dkz;//0.0;//
			}
			if(iz>NZ/2)
			{
				kz=2*pi*(NZ-1-iz)*dkz;//2*pi*(iz-NZ/2)*dkz;//0.0;//
			}

			ip=ix*NZ+iz;

			pp[ip].x=pp[ip].x*(kx*kx+kz*kz);
			pp[ip].y=pp[ip].y*(kx*kx+kz*kz);

		}
	} 

	// fft(r_real,r_imag,NFFT,-1);

	cudaMemcpy(temp,pp,sizeof(cufftComplex)*NX*NZ,cudaMemcpyHostToDevice);
	cufftExecC2C(plan,temp,tempout,CUFFT_INVERSE);
	cudaMemcpy(pp,tempout,sizeof(cufftComplex)*NX*NZ,cudaMemcpyDeviceToHost);

	for(ix=0;ix<nxx;ix++)
	{            
		for(iz=0;iz<nzz;iz++)
		{
			image[ix*nzz+iz]=pp[ix*NZ+iz].x;
		}
	} 

	cudaFreeHost(pp);
	cudaFree(temp);
	cudaFree(tempout);
	cufftDestroy(plan);

	return;
}

extern "C"
void congpu_fre(float *seismogram_syn, float *seismogram_obs, float *seismogram_rms, float *Misfit, int i, 
		float *ref_obs, float *ref_syn, int itmax, int nx, int s_ix, int *r_ix, int pml)
{ 
	cudaSetDevice(i);

	int ix,it,ip,K,NX;
	float epsilon,rms,rmsmax;

	K=(int)ceil(log(1.0*itmax)/log(2.0));
	NX=(int)pow(2.0,K);	

	int reft=s_ix-pml+1;

	int BATCH=nx;
	int NTP=NX*BATCH;

	if((nx-BATCH)!=0)
	{
		printf("CUFFT BREAK DOWN !! \n");
		return;
	}

	cufftComplex *xx,*d,*h,*sh,*r,*ms,*rr,*temp,*temp1,*obs;

	cudaMallocHost((void **)&xx, sizeof(cufftComplex)*NX*BATCH);
	cudaMallocHost((void **)&d, sizeof(cufftComplex)*NX*BATCH);
	cudaMallocHost((void **)&h, sizeof(cufftComplex)*NX);
	cudaMallocHost((void **)&sh,sizeof(cufftComplex)*NX);
	cudaMallocHost((void **)&r, sizeof(cufftComplex)*NX*BATCH);
	cudaMallocHost((void **)&ms,sizeof(cufftComplex)*NX*BATCH);
	cudaMallocHost((void **)&rr,sizeof(cufftComplex)*NX*BATCH);
	cudaMallocHost((void **)&obs,sizeof(cufftComplex)*NX*BATCH);

	cudaMalloc((void **)&temp1,sizeof(cufftComplex)*NX);
	cudaMalloc((void **)&temp,sizeof(cufftComplex)*NX*BATCH);

	for(it=0;it<NX;it++)
	{ 
		h[it].x=0.0;
		h[it].y=0.0; 

		sh[it].x=0.0;
		sh[it].y=0.0;    
	}

	for(it=0;it<itmax;it++)
	{
		h[it].x = ref_obs[it];
		sh[it].x= ref_syn[it];  
	}

	cufftHandle plan1,plan;
	cufftPlan1d(&plan1,NX,CUFFT_C2C,1);
	cufftPlan1d(&plan,NX,CUFFT_C2C,BATCH);

	cudaMemcpy(temp1,h,sizeof(cufftComplex)*NX,cudaMemcpyHostToDevice);
	cufftExecC2C(plan1,temp1,temp1,CUFFT_FORWARD);
	cudaMemcpy(h,temp1,sizeof(cufftComplex)*NX,cudaMemcpyDeviceToHost);

	cudaMemcpy(temp1,sh,sizeof(cufftComplex)*NX,cudaMemcpyHostToDevice);
	cufftExecC2C(plan1,temp1,temp1,CUFFT_FORWARD);
	cudaMemcpy(sh,temp1,sizeof(cufftComplex)*NX,cudaMemcpyDeviceToHost);  	

	//    for(ix=0;ix<nx;ix++)
	{
		for(it=0;it<NTP;it++)
		{ 
			xx[it].x=0.0;
			xx[it].y=0.0; 
			d[it].x=0.0;
			d[it].y=0.0;   
			r[it].x=0.0;
			r[it].y=0.0;  
		}            
		for(ix=0;ix<nx;ix++)
		{
			for(it=0;it<itmax;it++)
			{
				xx[ix*NX+it].x=seismogram_syn[ix*itmax+it];
				d[ix*NX+it].x=seismogram_obs[ix*itmax+it];	            
			}
		}   

		cudaMemcpy(temp,xx,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
		cufftExecC2C(plan,temp,temp,CUFFT_FORWARD);
		cudaMemcpy(xx,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

		cudaMemcpy(temp,d,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
		cufftExecC2C(plan,temp,temp,CUFFT_FORWARD);
		cudaMemcpy(d,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

		for(it=0;it<NTP;it++)
		{
			obs[it].x=sh[it%NX].x*d[it].x-sh[it%NX].y*d[it].y;
			obs[it].y=sh[it%NX].x*d[it].y+sh[it%NX].y*d[it].x;
			r[it].x=xx[it].x*h[it%NX].x-xx[it].y*h[it%NX].y-obs[it].x;
			r[it].y=xx[it].x*h[it%NX].y+xx[it].y*h[it%NX].x-obs[it].y;
		}  

		// fft(r_real,r_imag,NFFT,-1);

		cudaMemcpy(temp,r,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
		cufftExecC2C(plan,temp,temp,CUFFT_INVERSE);
		cudaMemcpy(ms,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

		cudaMemcpy(temp,obs,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
		cufftExecC2C(plan,temp,temp,CUFFT_INVERSE);
		cudaMemcpy(obs,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

		epsilon=0.0;
		rms=0.0;
		for(ix=0;ix<nx;ix++)
		{
			for(it=0;it<itmax;it++)
			{
				epsilon+=2*fabs(obs[ix*NX+it].x)/(itmax*nx);
			}
		}
		for(ix=0;ix<nx;ix++)
		{
			for(it=0;it<itmax;it++)
			{
				rms=rms+ms[ix*NX+it].x*ms[ix*NX+it].x/(epsilon*epsilon);
			}
		}
		*Misfit+=sqrt(1+rms)-1;

		// Calculate the r of ( f= rXdref )	Right hide term of adjoint equation!!!
		for(it=0;it<NTP;it++)
		{
			ms[it].x=ms[it].x/(epsilon*epsilon*sqrt(1+ms[it].x*ms[it].x/(epsilon*epsilon)));  //Time domain ms.x==u*dref-vref*d
			ms[it].y=0.0;
		}

		cudaMemcpy(temp,ms,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
		cufftExecC2C(plan,temp,temp,CUFFT_FORWARD);
		cudaMemcpy(r,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

		for(it=0;it<NTP;it++)
		{
			rr[it].x=h[it%NX].x*r[it].x+h[it%NX].y*r[it].y;

			rr[it].y=h[it%NX].x*r[it].y-h[it%NX].y*r[it].x;
		}   

		//fft(rr_real,rr_imag,NX*BATCH,-1);

		cudaMemcpy(temp,rr,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
		cufftExecC2C(plan,temp,temp,CUFFT_INVERSE);
		cudaMemcpy(rr,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

		rmsmax=0.0;
		for(ix=0;ix<nx;ix++)
		{
			for(it=0;it<itmax;it++)
			{
				ip=ix*itmax+it;
				seismogram_rms[ip]=rr[ix*NX+it].x;
				if(rmsmax<fabs(seismogram_rms[ip]))
				{
					rmsmax=fabs(seismogram_rms[ip]);
				}
	//			if(it>=itmax-60)
				{
	//				seismogram_rms[ip]=0.0;
				}            
			}
		}

		for(it=0;it<itmax*nx;it++)
		{
			seismogram_rms[it]/=rmsmax;
		}
	}

	cudaFreeHost(xx);
	cudaFreeHost(d);
	cudaFreeHost(h);
	cudaFreeHost(sh);
	cudaFreeHost(r);   
	cudaFreeHost(ms);   
	cudaFreeHost(rr);
	cudaFreeHost(obs); 

	cudaFree(temp);
	cudaFree(temp1);

	cufftDestroy(plan1);
	cufftDestroy(plan);

	return;
}

extern "C"
void hilbert_transform1d(float *rick, float *rick_hilbert, int itmax)
{
	int it,ip,K,NZ;
	float epsilon,rms,rmsmax;

	K=(int)ceil(log(1.0*itmax)/log(2.0));
	NZ=(int)pow(2.0,K);	

	cufftComplex *rickf,*rickf_hilbert,*temp;

	cudaMallocHost((void **)&rickf, sizeof(cufftComplex)*NZ);
	cudaMallocHost((void **)&rickf_hilbert, sizeof(cufftComplex)*NZ);

	cudaMalloc((void **)&temp,sizeof(cufftComplex)*NZ);

	for(it=0;it<NZ;it++)
	{ 
		rickf[it].x=0.0;
		rickf[it].y=0.0; 

		rickf_hilbert[it].x=0.0;
		rickf_hilbert[it].y=0.0; 
	}

	cufftHandle plan;
	cufftPlan1d(&plan,NZ,CUFFT_C2C,1);

	for(it=0;it<itmax;it++)
	{
		rickf[it].x=rick[it];
	}

	cudaMemcpy(temp,rickf,sizeof(cufftComplex)*NZ,cudaMemcpyHostToDevice);
	cufftExecC2C(plan,temp,temp,CUFFT_FORWARD);
	cudaMemcpy(rickf,temp,sizeof(cufftComplex)*NZ,cudaMemcpyDeviceToHost);

	for(it=1;it<ceil(NZ/2);it++)
	{
		rickf_hilbert[it].x= rickf[it].y;
		rickf_hilbert[it].y=-rickf[it].x;
	}  
	for(it=ceil(NZ/2);it<NZ;it++)
	{
		rickf_hilbert[it].x=-rickf[it].y;
		rickf_hilbert[it].y= rickf[it].x;
	}  

	// fft(r_real,r_imag,NFFT,-1);

	cudaMemcpy(temp,rickf_hilbert,sizeof(cufftComplex)*NZ,cudaMemcpyHostToDevice);
	cufftExecC2C(plan,temp,temp,CUFFT_INVERSE);
	cudaMemcpy(rickf_hilbert,temp,sizeof(cufftComplex)*NZ,cudaMemcpyDeviceToHost);

	for(it=0;it<itmax;it++)
	{
		rick_hilbert[it]=rickf_hilbert[it].x/NZ;
	}

	cudaFreeHost(rickf);
	cudaFreeHost(rickf_hilbert);

	cudaFree(temp);

	cufftDestroy(plan);

	return;
}

extern "C"
void hilbert_transform2d(float *data, float *data_hilbert, int i, 
		int nz, int nx)
{
	cudaSetDevice(i);

	int ix,iz,ip,K,NZ;
	float epsilon,rms,rmsmax;

	K=(int)ceil(log(1.0*nz)/log(2.0));
	NZ=(int)pow(2.0,K);

	int BATCH=nx;
	int NTP=NZ*BATCH;

	if((nx-BATCH)!=0)
	{
		printf("CUFFT BREAK DOWN !! \n");
		return;
	}

	cufftComplex *dataf,*dataf_hilbert,*temp;

	cudaMallocHost((void **)&dataf, sizeof(cufftComplex)*NZ*BATCH);
	cudaMallocHost((void **)&dataf_hilbert, sizeof(cufftComplex)*NZ*BATCH);

	cudaMalloc((void **)&temp,sizeof(cufftComplex)*NZ*BATCH);

	for(ip=0;ip<NTP;ip++)
	{
		dataf[ip].x=0.0;
		dataf[ip].y=0.0; 

		dataf_hilbert[ip].x=0.0;
		dataf_hilbert[ip].y=0.0;    
	}

	cufftHandle plan;
	cufftPlan1d(&plan,NZ,CUFFT_C2C,BATCH);

	for(ix=0;ix<nx;ix++)
	{
		for(iz=0;iz<nz;iz++)
		{
			dataf[ix*NZ+iz].x=data[ix*nz+iz];
		}
	}

	cudaMemcpy(temp,dataf,sizeof(cufftComplex)*NZ*BATCH,cudaMemcpyHostToDevice);
	cufftExecC2C(plan,temp,temp,CUFFT_FORWARD);
	cudaMemcpy(dataf,temp,sizeof(cufftComplex)*NZ*BATCH,cudaMemcpyDeviceToHost);

	for(ix=0;ix<nx;ix++)
	{
		for(iz=1;iz<ceil(NZ/2);iz++)
		{
			ip=ix*NZ+iz;

			dataf_hilbert[ip].x= dataf[ip].y;
			dataf_hilbert[ip].y=-dataf[ip].x;
		}  
		for(iz=ceil(NZ/2);iz<NZ;iz++)
		{
			ip=ix*NZ+iz;

			dataf_hilbert[ip].x=-dataf[ip].y;
			dataf_hilbert[ip].y= dataf[ip].x;
		}  

	}  

	// fft(r_real,r_imag,NFFT,-1);

	cudaMemcpy(temp,dataf_hilbert,sizeof(cufftComplex)*NZ*BATCH,cudaMemcpyHostToDevice);
	cufftExecC2C(plan,temp,temp,CUFFT_INVERSE);
	cudaMemcpy(dataf_hilbert,temp,sizeof(cufftComplex)*NZ*BATCH,cudaMemcpyDeviceToHost);

	for(ix=0;ix<nx;ix++)
	{
		for(iz=0;iz<nz;iz++)
		{
			ip=ix*nz+iz;
			data_hilbert[ip]=dataf_hilbert[ix*NZ+iz].x/NZ;
		}
	}

	cudaFreeHost(dataf);
	cudaFreeHost(dataf_hilbert);

	cudaFree(temp);

	cufftDestroy(plan);

	return;
}

/*=============================================
 * Allocate the memory for wavefield simulation
 * ===========================================*/
extern "C"
void variables_malloc(int ntx, int ntz, int ntp, int nx, int nz,
		int pml, int Lc, float dx, float dz, int itmax, int thetan, int delt,
		struct MultiGPU plan[], int GPU_N, int rnmax
		)
{
	int i;

	size_t size_model=sizeof(float)*ntp;

	int KZ=(int)ceil(log(1.0*ntz)/log(2.0));
	int nfftz=pow(2.0,KZ);

	// ==========================================================
	// allocate the memory for the device
	// allocate the memory for the device
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		// allocate the memory of vx,vy,vz,sigmaxx,sigmayy,...
		plan[i].vx=(float*)malloc(sizeof(float)*ntp); 
		plan[i].vz=(float*)malloc(sizeof(float)*ntp); 
		//plan[i].p=(float*)malloc(sizeof(float)*ntp);
		cudaMallocHost((void**)&plan[i].p,sizeof(cufftComplex)*ntp);
		cudaMallocHost((void**)&plan[i].htp,sizeof(cufftComplex)*ntp);
		cudaMallocHost((void**)&plan[i].spu,sizeof(cufftComplex)*ntx*nfftz);
		cudaMallocHost((void**)&plan[i].spd,sizeof(cufftComplex)*ntx*nfftz);
		cudaMallocHost((void**)&plan[i].rpu,sizeof(cufftComplex)*ntx*nfftz);
		cudaMallocHost((void**)&plan[i].rpd,sizeof(cufftComplex)*ntx*nfftz);

		///device//////////
		///device//////////
		///device//////////

		cudaMalloc((void**)&plan[i].d_fxz,delt*size_model);
		cudaMalloc((void**)&plan[i].d_fxzr,delt*size_model);

		cudaMalloc((void**)&plan[i].d_seismogram,sizeof(float)*itmax*rnmax);
		cudaMalloc((void**)&plan[i].d_seismogram_up,sizeof(float)*itmax*rnmax);
		cudaMalloc((void**)&plan[i].d_seismogram_down,sizeof(float)*itmax*rnmax);
		cudaMalloc((void**)&plan[i].d_seismogram_rms,sizeof(float)*itmax*(rnmax));
		cudaMalloc((void**)&plan[i].d_seismogram_hrms,sizeof(float)*itmax*(rnmax));

		cudaMalloc((void**)&plan[i].d_coef,sizeof(float)*nfftz);
		cudaMalloc((void**)&plan[i].d_r_iz,sizeof(int)*rnmax);

		cudaMalloc((void**)&plan[i].d_rick,sizeof(float)*itmax);        // ricker wave 
		cudaMalloc((void**)&plan[i].d_hrick,sizeof(float)*itmax);        // ricker wave 
		cudaMalloc((void**)&plan[i].d_rc,sizeof(float)*Lc);        // ricker wave 
//		cudaMalloc((void**)&plan[i].d_asr,sizeof(float)*NN);        // ricker wave 

		cudaMalloc((void**)&plan[i].d_vp,size_model);
		cudaMalloc((void**)&plan[i].d_rho,size_model);
		cudaMalloc((void**)&plan[i].d_mx,sizeof(int)*ntx);

		cudaMalloc((void**)&plan[i].d_a_x,sizeof(float)*ntx);
		cudaMalloc((void**)&plan[i].d_a_x_half,sizeof(float)*ntx);
		cudaMalloc((void**)&plan[i].d_a_z,sizeof(float)*ntz);
		cudaMalloc((void**)&plan[i].d_a_z_half,sizeof(float)*ntz);

		cudaMalloc((void**)&plan[i].d_b_x,sizeof(float)*ntx);
		cudaMalloc((void**)&plan[i].d_b_x_half,sizeof(float)*ntx);
		cudaMalloc((void**)&plan[i].d_b_z,sizeof(float)*ntz);
		cudaMalloc((void**)&plan[i].d_b_z_half,sizeof(float)*ntz);      // atten parameters

		cudaMalloc((void**)&plan[i].d_image_ppdu,size_model);
		cudaMalloc((void**)&plan[i].d_image_ppud,size_model);
		cudaMalloc((void**)&plan[i].d_image_ppcig,size_model*thetan);

		cudaMalloc((void**)&plan[i].d_image_sources,size_model);
		cudaMalloc((void**)&plan[i].d_image_receivers,size_model);

		cudaMalloc((void**)&plan[i].d_vx,size_model);
		cudaMalloc((void**)&plan[i].d_vz,size_model);
		//cudaMalloc((void**)&plan[i].d_p,size_model);
		cudaMalloc((void**)&plan[i].d_p,sizeof(cufftComplex)*ntp);

		cudaMalloc((void**)&plan[i].d_htvx,size_model);
		cudaMalloc((void**)&plan[i].d_htvz,size_model);
		//cudaMalloc((void**)&plan[i].d_htp,size_model);
		cudaMalloc((void**)&plan[i].d_htp,sizeof(cufftComplex)*ntp);

		////////////////////////////////////////
		cudaMalloc((void**)&plan[i].d_hp,sizeof(cufftComplex)*ntx*ntz);
		cudaMalloc((void**)&plan[i].d_hhtp,sizeof(cufftComplex)*ntx*ntz);
		////////////////////////////////////////
		////////// Seperated Wavefield//////////
		cudaMalloc((void**)&plan[i].d_temp,sizeof(cufftComplex)*ntx*nfftz);
		cudaMalloc((void**)&plan[i].d_spu,sizeof(cufftComplex)*ntx*nfftz);
		cudaMalloc((void**)&plan[i].d_spd,sizeof(cufftComplex)*ntx*nfftz);
		cudaMalloc((void**)&plan[i].d_rpu,sizeof(cufftComplex)*ntx*nfftz);
		cudaMalloc((void**)&plan[i].d_rpd,sizeof(cufftComplex)*ntx*nfftz);

		////////////////////////////////////////

		cudaMalloc((void**)&plan[i].dp_dt,size_model);

		cudaMalloc((void**)&plan[i].d_vx_inv,size_model);
		cudaMalloc((void**)&plan[i].d_vz_inv,size_model);
		//cudaMalloc((void**)&plan[i].d_p_inv,size_model);
		cudaMalloc((void**)&plan[i].d_p_inv,sizeof(cufftComplex)*ntp);

		cudaMalloc((void**)&plan[i].d_htvx_inv,size_model);
		cudaMalloc((void**)&plan[i].d_htvz_inv,size_model);
		//cudaMalloc((void**)&plan[i].d_htp_inv,size_model);
		cudaMalloc((void**)&plan[i].d_htp_inv,sizeof(cufftComplex)*ntp);

		cudaMalloc((void**)&plan[i].d_phi_vx_x,size_model);
		cudaMalloc((void**)&plan[i].d_phi_vz_z,size_model);

		cudaMalloc((void**)&plan[i].d_phi_htvx_x,size_model);
		cudaMalloc((void**)&plan[i].d_phi_htvz_z,size_model);

		cudaMalloc((void**)&plan[i].d_phi_p_x,size_model);
		cudaMalloc((void**)&plan[i].d_phi_p_z,size_model);

		cudaMalloc((void**)&plan[i].d_phi_htp_x,size_model);
		cudaMalloc((void**)&plan[i].d_phi_htp_z,size_model);

		cudaMalloc((void**)&plan[i].d_p_borders_up,sizeof(float)*Lc*itmax*nx);
		cudaMalloc((void**)&plan[i].d_p_borders_bottom,sizeof(float)*Lc*itmax*nx);
		cudaMalloc((void**)&plan[i].d_p_borders_left,sizeof(float)*Lc*itmax*(nz-2*Lc));
		cudaMalloc((void**)&plan[i].d_p_borders_right,sizeof(float)*Lc*itmax*(nz-2*Lc));

		cudaMalloc((void**)&plan[i].d_htp_borders_up,sizeof(float)*Lc*itmax*nx);
		cudaMalloc((void**)&plan[i].d_htp_borders_bottom,sizeof(float)*Lc*itmax*nx);
		cudaMalloc((void**)&plan[i].d_htp_borders_left,sizeof(float)*Lc*itmax*(nz-2*Lc));
		cudaMalloc((void**)&plan[i].d_htp_borders_right,sizeof(float)*Lc*itmax*(nz-2*Lc));
		//
		/////////////////////////////////
		cufftPlan1d(&plan[i].forward_z, nfftz,CUFFT_C2C,ntx);
		cufftPlan1d(&plan[i].backward_z,nfftz,CUFFT_C2C,ntx);

	}
}

/*=============================================
 * Free the memory for wavefield simulation
 * ===========================================*/
extern "C"
void variables_free(int ntx, int ntz, int ntp, int nx, int nz,
		int pml, int Lc, float dx, float dz, int itmax,
		struct MultiGPU plan[], int GPU_N, int rnmax
		)
{
	int i;
	//free the memory of DEVICE
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		//free the memory of Vx,Vy,Vz,Sigmaxx,Sigmayy,Sigmazz...  
		free(plan[i].vx);
		free(plan[i].vz);
		cudaFree(plan[i].p);
		cudaFree(plan[i].htp);
		cudaFree(plan[i].spu);
		cudaFree(plan[i].spd);
		cudaFree(plan[i].rpu);
		cudaFree(plan[i].rpd);

		////device////
		////device////
		////device////

		cudaFree(plan[i].d_fxz);
		cudaFree(plan[i].d_fxzr);

		cudaFree(plan[i].d_seismogram);
		cudaFree(plan[i].d_seismogram_up);
		cudaFree(plan[i].d_seismogram_down);
		cudaFree(plan[i].d_seismogram_rms);
		cudaFree(plan[i].d_seismogram_hrms);

		cudaFree(plan[i].d_coef);
		cudaFree(plan[i].d_r_iz);

		cudaFree(plan[i].d_rick);
		cudaFree(plan[i].d_hrick);
		cudaFree(plan[i].d_rc);
//		cudaFree(plan[i].d_asr);

		cudaFree(plan[i].d_vp);
		cudaFree(plan[i].d_rho);
		cudaFree(plan[i].d_mx);

		cudaFree(plan[i].d_a_x);
		cudaFree(plan[i].d_a_x_half);
		cudaFree(plan[i].d_a_z);
		cudaFree(plan[i].d_a_z_half);

		cudaFree(plan[i].d_b_x);
		cudaFree(plan[i].d_b_x_half);
		cudaFree(plan[i].d_b_z);
		cudaFree(plan[i].d_b_z_half);

		cudaFree(plan[i].d_image_ppdu);
		cudaFree(plan[i].d_image_ppud);
		cudaFree(plan[i].d_image_ppcig);

		cudaFree(plan[i].d_image_sources);
		cudaFree(plan[i].d_image_receivers);

		cudaFree(plan[i].d_vx);
		cudaFree(plan[i].d_vz);
		cudaFree(plan[i].d_p);

		cudaFree(plan[i].d_htvx);
		cudaFree(plan[i].d_htvz);
		cudaFree(plan[i].d_htp);

		//////////////////////////
		cudaFree(plan[i].d_hp);
		cudaFree(plan[i].d_temp);
		cudaFree(plan[i].d_hhtp);
		//////////////////////////
		////////////////////////////////////////
		////////// Seperated Wavefield//////////
		cudaFree(plan[i].d_spu);
		cudaFree(plan[i].d_spd);
		cudaFree(plan[i].d_rpu);
		cudaFree(plan[i].d_rpd);

		cudaFree(plan[i].dp_dt);

		cudaFree(plan[i].d_vx_inv);
		cudaFree(plan[i].d_vz_inv);
		cudaFree(plan[i].d_p_inv);

		cudaFree(plan[i].d_htvx_inv);
		cudaFree(plan[i].d_htvz_inv);
		cudaFree(plan[i].d_htp_inv);

		cudaFree(plan[i].d_phi_vx_x);
		cudaFree(plan[i].d_phi_vz_z);

		cudaFree(plan[i].d_phi_htvx_x);
		cudaFree(plan[i].d_phi_htvz_z);

		cudaFree(plan[i].d_phi_p_x);
		cudaFree(plan[i].d_phi_p_z);

		cudaFree(plan[i].d_phi_htp_x);
		cudaFree(plan[i].d_phi_htp_z);

		cudaFree(plan[i].d_p_borders_up);
		cudaFree(plan[i].d_p_borders_bottom);
		cudaFree(plan[i].d_p_borders_left);
		cudaFree(plan[i].d_p_borders_right);

		cudaFree(plan[i].d_htp_borders_up);
		cudaFree(plan[i].d_htp_borders_bottom);
		cudaFree(plan[i].d_htp_borders_left);
		cudaFree(plan[i].d_htp_borders_right);

		///////////////////////////////////////
		cufftDestroy(plan[i].forward_z);
		cufftDestroy(plan[i].backward_z);
	}
}

extern "C"
void getdevice(int *GPU_N)
{
	
	cudaGetDeviceCount(GPU_N);	
//	GPU_N=6;//4;//2;//
}

