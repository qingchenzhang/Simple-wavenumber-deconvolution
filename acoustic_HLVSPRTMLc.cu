#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define BLOCK_SIZE 16

#define OUTPUT_SNAP 0
#define PI 3.1415926

#include "headrtm.h"
#include "cufft.h"

//#define BATCH 834

#define reference_window 1 //0--without 1--with

struct Multistream
{
	cudaStream_t stream,stream_back;
};

__global__ void fdtd_cpml_2d_GPU_kernel_vx(
		float *rho, int *mx, int itmax,
		float *a_x_half, float *a_z, 
		float *b_x_half, float *b_z, 
		float *vx, float *p,
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
			dp_dx+=rc[ic]*(p[ip+(ic+1)*ntz]-p[ip-ic*ntz])*one_over_dx;
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
		float *vz, float *p, 
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
			dp_dz+=rc[ic]*(p[ip+(ic+1)]-p[ip-ic])*one_over_dz;
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
		float *vx, float *vz, float *p,
		float *phi_vx_x, float *phi_vz_z, 
		int ntp, int ntx, int ntz,
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

		p[ip]=dt*rho[ip]*vp[ip]*vp[ip]*(dvx_dx+phi_vx_x[ip]+dvz_dz+phi_vz_z[ip])+p[ip];

	}

	if(iz==s_iz&&mx[ix]==s_ix)
	{
		p[ip]+=rick[it];
	}

	// Seismogram...   
	//if(ix==r_ix)
	if(ix==pmlc)
		for(ii=0;ii<r_n/2;ii++)
		{
			if(iz==r_iz[ii])
			{
				seismogram[ii*itmax+it]=p[ip];
			}
		}
	if(ix==ntx-pmlc-1)
		for(ii=0;ii<r_n/2;ii++)
		{
			if(iz==r_iz[ii])
			{
				seismogram[(ii+r_n/2)*itmax+it]=p[ip];
			}
		}

	// Borders...
	if(inv_flag==1)
	{
		if(ix>=pmlc&&ix<=ntx-pmlc-1&&iz>=pmlc&&iz<=pmlc+Lc-1)
		{
			p_borders_up[(iz-pmlc)*itmax*nx+it*nx+ix-pmlc]=p[ip];
		}
		if(ix>=pmlc&&ix<=ntx-pmlc-1&&iz>=ntz-pmlc-Lc&&iz<=ntz-pmlc-1)
		{
			p_borders_bottom[(iz-ntz+pmlc+Lc)*itmax*nx+it*nx+ix-pmlc]=p[ip];
		}

		if(iz>=pmlc+Lc&&iz<=ntz-pmlc-Lc-1&&ix>=pmlc&&ix<=pmlc+Lc-1)
		{
			p_borders_left[(ix-pmlc)*itmax*(nz-2*Lc)+it*(nz-2*Lc)+iz-pmlc-Lc]=p[ip];
		}
		if(iz>=pmlc+Lc&&iz<=ntz-pmlc-Lc-1&&ix>=ntx-pmlc-Lc&&ix<=ntx-pmlc-1)
		{
			p_borders_right[(ix-ntx+pmlc+Lc)*itmax*(nz-2*Lc)+it*(nz-2*Lc)+iz-pmlc-Lc]=p[ip];
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
		float *rick, int itmax, float dt, int myid,
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
	int ix,iz;
	int pmlc=pml+Lc;

	size_t size_model=sizeof(float)*ntp;

	FILE *fp;
	char filename[40];

	Multistream plans[GPU_N];

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		cudaStreamCreate(&plans[i].stream);	
	}

	///////////////////////////////
	// initialize the fields........................

	// copy the vectors from the host to the device
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		cudaMemcpyAsync(plan[i].d_r_iz,ss[is+i].r_iz,sizeof(int)*ss[is+i].r_n,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_rick,rick,sizeof(float)*itmax,cudaMemcpyHostToDevice,plans[i].stream);
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
		cudaMemsetAsync(plan[i].d_p,0.0,size_model,plans[i].stream);

		cudaMemsetAsync(plan[i].d_phi_vx_x,0.0,size_model,plans[i].stream);
		cudaMemsetAsync(plan[i].d_phi_vz_z,0.0,size_model,plans[i].stream);

		cudaMemsetAsync(plan[i].d_phi_p_x,0.0,size_model,plans[i].stream);
		cudaMemsetAsync(plan[i].d_phi_p_z,0.0,size_model,plans[i].stream);
	}
	/////////////////////////////////
	// =============================================================================

	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 dimGrid((ntx+dimBlock.x-1)/dimBlock.x,(ntz+dimBlock.y-1)/dimBlock.y);

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

			fdtd_cpml_2d_GPU_kernel_p<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rick, plan[i].d_vp, plan[i].d_rho, plan[i].d_mx,
				 plan[i].d_a_x, plan[i].d_a_z, plan[i].d_b_x, plan[i].d_b_z,
				 plan[i].d_vx, plan[i].d_vz, plan[i].d_p,
				 plan[i].d_phi_vx_x, plan[i].d_phi_vz_z,
				 ntp, ntx, ntz,
				 plan[i].d_seismogram, ss[is+i].r_ix, plan[i].d_r_iz, ss[is+i].r_n,
				 pml, Lc, plan[i].d_rc, dx, dz, dt,
				 ss[is+i].s_ix, ss[is+i].s_iz, it,
				 inv_flag, itmax,
				 plan[i].d_p_borders_up, plan[i].d_p_borders_bottom,
				 plan[i].d_p_borders_left, plan[i].d_p_borders_right
				);
/*
			if(inv_flag==1&&it%100==0)
			{
				cudaMemcpyAsync(plan[i].vx,plan[i].d_p,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
				cudaStreamSynchronize(plans[i].stream);

				sprintf(filename,"./output/%dvx%d.dat",it,i);     
				fp=fopen(filename,"wb");
				for(ix=pmlc;ix<ntx-pmlc;ix++)
				{
					for(iz=pmlc;iz<ntz-pmlc;iz++)
					{					
						fwrite(&plan[i].vx[ix*ntz+iz],sizeof(float),1,fp);
					}
				}
				fclose(fp);
			}
*/

		}//end GPU_N
	}//end it

	for(i=0;i<GPU_N;i++)
	{
		if(inv_flag==0)
		{
			cudaMemcpyAsync(plan[i].seismogram_obs,plan[i].d_seismogram,
					sizeof(float)*ss[is+i].r_n*itmax,cudaMemcpyDeviceToHost,plans[i].stream);
		}
		else
		{
			cudaMemcpyAsync(plan[i].seismogram_syn,plan[i].d_seismogram,
					sizeof(float)*ss[is+i].r_n*itmax,cudaMemcpyDeviceToHost,plans[i].stream);
		}

		if(inv_flag==1)
		{
/*			cudaMemcpyAsync(plan[i].p_borders_up,plan[i].d_p_borders_up,
					sizeof(float)*Lc*nx*itmax,cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(plan[i].p_borders_bottom,plan[i].d_p_borders_bottom,
					sizeof(float)*Lc*nx*itmax,cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(plan[i].p_borders_left,plan[i].d_p_borders_left,
					sizeof(float)*Lc*(nz-2*Lc)*itmax,cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(plan[i].p_borders_right,plan[i].d_p_borders_right,
					sizeof(float)*Lc*(nz-2*Lc)*itmax,cudaMemcpyDeviceToHost,plans[i].stream);
*/
			// Output The wavefields when Time=Itmax;

			cudaMemcpyAsync(plan[i].vx,plan[i].d_vx,size_model,cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(plan[i].vz,plan[i].d_vz,size_model,cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(plan[i].p,plan[i].d_p,size_model,cudaMemcpyDeviceToHost,plans[i].stream);

			cudaStreamSynchronize(plans[i].stream);

			sprintf(filename,"./output/wavefield%d_itmax%d.dat",myid,i);
			fp=fopen(filename,"wb");
			fwrite(&plan[i].vx[0],sizeof(float),ntp,fp);
			fwrite(&plan[i].vz[0],sizeof(float),ntp,fp);

			fwrite(&plan[i].p[0],sizeof(float),ntp,fp);
			fclose(fp);
		}
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

__global__ void fdtd_2d_GPU_kernel_p_backward(
		float *rick, float *vp, float *rho, int *mx,
		float *vx, float *vz, float *p,
		int ntp, int ntx, int ntz, int pml, int Lc, float *rc,
		float dx, float dz, float dt, int s_ix,
		int s_iz, int it, float *dp_dt
		)

{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int pmlc=pml+Lc;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dz;

	float dvx_dx,dvz_dz;
	int ic,ii;
	int ip=ix*ntz+iz;

	if(iz>=pmlc&&iz<=ntz-pmlc-1&&ix>=pmlc&&ix<=ntx-pmlc-1)
	{
		dvx_dx=0.0;
		dvz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dvx_dx+=rc[ic]*(vx[ip+ic*ntz]-vx[ip-(ic+1)*ntz])*one_over_dx;
			dvz_dz+=rc[ic]*(vz[ip+ic]-vz[ip-(ic+1)])*one_over_dz;
		}

		p[ip]=dt*rho[ip]*vp[ip]*vp[ip]*(dvx_dx+dvz_dz)+p[ip];

		dp_dt[ip]=rho[ip]*vp[ip]*vp[ip]*(dvx_dx+dvz_dz);
	}

	if(iz==s_iz&&mx[ix]==s_ix)
	{
		p[ip]-=rick[it+1];
	}
	
	__syncthreads();

}


__global__ void fdtd_2d_GPU_kernel_vx_backward(
		float *rho, int *mx,
		float *vx, float *p,
		int ntp, int ntx, int ntz, int pml, int Lc, float *rc,
		float dx, float dz, float dt
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int pmlc=pml+Lc;

	int ic;
	int ip=ix*ntz+iz;

	float one_over_dx=1.0/dx;

	float dp_dx,one_over_rho_half_x;

	if(iz>=pmlc&&iz<=ntz-pmlc-1&&ix>=pmlc&&ix<=ntx-pmlc-1)
	{
		dp_dx=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dp_dx+=rc[ic]*(p[ip+(ic+1)*ntz]-p[ip-ic*ntz])*one_over_dx;
		}

		one_over_rho_half_x=1/(0.5*(rho[ip]+rho[ip+1*ntz]));

		vx[ip]=dt*one_over_rho_half_x*dp_dx+vx[ip];
	}

	__syncthreads();
}

__global__ void fdtd_2d_GPU_kernel_vz_backward(
		float *rho, int *mx,
		float *vz, float *p,
		int ntp, int ntx, int ntz, int pml, int Lc, float *rc,
		float dx, float dz, float dt
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

	int pmlc=pml+Lc;

	int ic;
	int ip=ix*ntz+iz;

	if(iz>=pmlc&&iz<=ntz-pmlc-1&&ix>=pmlc&&ix<=ntx-pmlc-1)
	{
		dp_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dp_dz+=rc[ic]*(p[ip+(ic+1)]-p[ip-ic])*one_over_dz;
		}

		one_over_rho_half_z=1/(0.5*(rho[ip]+rho[ip+1]));

		vz[ip]=dt*one_over_rho_half_z*dp_dz+vz[ip];
	}

	__syncthreads();

}

__global__ void fdtd_2d_GPU_kernel_borders_backward
(
 float *p,
 float *p_borders_up, float *p_borders_bottom,
 float *p_borders_left, float *p_borders_right,
 int ntp, int ntx, int ntz, int pml, int Lc, float *rc, int it, int itmax
 )
{


	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int pmlc=pml+Lc;

	int ip=ix*ntz+iz;
	int nx=ntx-2*pmlc;
	int nz=ntz-2*pmlc;

	if(ix>=pmlc&&ix<=ntx-pmlc-1&&iz>=pmlc&&iz<=pmlc+Lc-1)
	{
		p[ip]=p_borders_up[(iz-pmlc)*itmax*nx+it*nx+ix-pmlc];
	}
	if(ix>=pmlc&&ix<=ntx-pmlc-1&&iz>=ntz-pmlc-Lc&&iz<=ntz-pmlc-1)
	{
		p[ip]=p_borders_bottom[(iz-ntz+pmlc+Lc)*itmax*nx+it*nx+ix-pmlc];
	}

	if(iz>=pmlc+Lc&&iz<=ntz-pmlc-Lc-1&&ix>=pmlc&&ix<=pmlc+Lc-1)
	{
		p[ip]=p_borders_left[(ix-pmlc)*itmax*(nz-2*Lc)+it*(nz-2*Lc)+iz-pmlc-Lc];
	}
	if(iz>=pmlc+Lc&&iz<=ntz-pmlc-Lc-1&&ix>=ntx-pmlc-Lc&&ix<=ntx-pmlc-1)
	{
		p[ip]=p_borders_right[(ix-ntx+pmlc+Lc)*itmax*(nz-2*Lc)+it*(nz-2*Lc)+iz-pmlc-Lc];
	}

	__syncthreads();

}


__global__ void fdtd_cpml_2d_GPU_kernel_vx_backward(
		float *rho, int *mx,
		float *a_x_half, float *a_z, 
		float *b_x_half, float *b_z, 
		float *vx, float *p,
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

	int ic,ip=ix*ntz+iz;

	float one_over_dx=1.0/dx;

	float dp_dx,one_over_rho_half_x;

	if(iz>=0&&iz<=ntz-1&&ix>=Lc-1&&ix<=ntx-Lc-1)
	{
		dp_dx=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dp_dx+=rc[ic]*(p[ip+(ic+1)*ntz]-p[ip-ic*ntz])*one_over_dx;
		}

		phi_p_x[ip]=b_x_half[ix]*phi_p_x[ip]+a_x_half[ix]*dp_dx;

		one_over_rho_half_x=1/(0.5*(rho[ip]+rho[ip+1*ntz]));

		vx[ip]=dt*one_over_rho_half_x*(dp_dx+phi_p_x[ip])+vx[ip];
	}   

	__syncthreads();

}


__global__ void fdtd_cpml_2d_GPU_kernel_vz_backward(
		float *rho, int *mx,
		float *a_x, float *a_z_half,
		float *b_x, float *b_z_half,
		float *vz, float *p, 
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

	int ic,ip=ix*ntz+iz;

	if(iz>=Lc-1&&iz<=ntz-Lc-1&&ix>=0&&ix<=ntx-1)
	{
		dp_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dp_dz+=rc[ic]*(p[ip+(ic+1)]-p[ip-ic])*one_over_dz;
		}

		phi_p_z[ip]=b_z_half[iz]*phi_p_z[ip]+a_z_half[iz]*dp_dz;

		one_over_rho_half_z=1/(0.5*(rho[ip]+rho[ip+1]));

		vz[ip]=dt*one_over_rho_half_z*(dp_dz+phi_p_z[ip])+vz[ip];
	}

	__syncthreads();

}

__global__ void fdtd_cpml_2d_GPU_kernel_p_backward(
		float *vp, float *rho, int *mx,
		float *a_x, float *a_z,
		float *b_x, float *b_z,
		float *vx, float *vz, float *p,
		float *phi_vx_x, float *phi_vz_z, 
		int ntp, int ntx, int ntz, int pml, int Lc, float *rc,
		float *seismogram_rms, int r_ix, int *r_iz, int r_n, 
		float dx, float dz, float dt, int itmax, int it
		)

{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int pmlc=pml+Lc;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dz;

	float dvx_dx,dvz_dz;
	int ic,ii,ip=ix*ntz+iz;

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

		p[ip]=dt*rho[ip]*vp[ip]*vp[ip]*(dvx_dx+phi_vx_x[ip]+dvz_dz+phi_vz_z[ip])+p[ip];

	}

	//if(ix==r_ix)
	if(ix==pmlc)
		for(ii=0;ii<r_n/2;ii++)
		{
			if(iz==r_iz[ii])
			{
				p[ip]=seismogram_rms[ii*itmax+it];
			}
		}
	if(ix==ntx-pmlc-1)
		for(ii=0;ii<r_n/2;ii++)
		{
			if(iz==r_iz[ii])
			{
				p[ip]=seismogram_rms[(ii+r_n/2)*itmax+it];
			}
		}
	__syncthreads();
}


__global__ void sum_image_GPU_kernel_image
(
 float *vp, float *rho, int *mx,
 float *p_inv, float *dp_dt, float *vx_inv, float *vz_inv, float *p, float *vx, float *vz,
 float *image_pp, float *image_sources, float *image_receivers, int s_ix, int r_ix, 
 int ntx, int ntz, int thetan, int pml, int Lc, float *rc, float dx, float dz
 )
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int pmlc=pml+Lc;

	int ip=ix*ntz+iz;

	float dp_dx,dp_dz;

	if(iz>=pmlc&&iz<=ntz-pmlc-1&&ix>=pmlc&&ix<=ntx-pmlc-1)//&&((ix>=s_ix&&ix<r_ix)||(ix<=s_ix&&ix>r_ix))
	{
		//image_pp[ip]+=p_inv[ip]*p[ip];
		image_pp[ip]+=dp_dt[ip]*p[ip];

		image_sources[ip]+=dp_dt[ip]*dp_dt[ip];//p_inv[ip]*p_inv[ip];//
		image_receivers[ip]+=p[ip]*p[ip];        
	}
	__syncthreads();
}

__global__ void sum_image_GPU_kernel_imageCIG
(
 float *vp, float *rho, int *mx, float *fxz, float *fxzr,
 float *p_inv, float *dp_dt, float *vx_inv, float *vz_inv, float *p, float *vx, float *vz,
 float *image_pp, float *image_ppcig,
 int ntx, int ntz, int thetan, int delt, int pml, int Lc, float *rc, float dx, float dz, int it, int itmax
 )
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int pmlc=pml+Lc;
	int ntp=ntx*ntz;

	int ip=ix*ntz+iz;
	int ipp=ix*ntz*thetan+iz*thetan;

	float dp_dx,dp_dz;
	
	float fx,fz,fxr,fzr,pre,pst;// poynting vector

	float ftheta,tht;// angle
	float betaf;
	int   ii,itheta;

	if(it==itmax-2)
	{
		for(ii=0;ii<delt*ntp;ii++)
		{
			fxz[ii]=0.0;
			fxzr[ii]=0.0;
		}
	}

	if(iz>=pmlc&&iz<=ntz-pmlc-1&&ix>=pmlc&&ix<=ntx-pmlc-1)
	{
		//angle commom image gathers//
		//angle commom image gathers//
		fx=-p_inv[ip]*vx_inv[ip]; //poynting vector
		fz=-p_inv[ip]*vz_inv[ip]; //poynting vector

		fxr=-p[ip]*vx[ip]; //poynting vector
		fzr=-p[ip]*vz[ip]; //poynting vector

		fxz[(it%delt)*ntp+ip]=(fx*fxr+fz*fzr);
		fxzr[(it%delt)*ntp+ip]=sqrtf(fx*fx+fz*fz)*sqrtf(fxr*fxr+fzr*fzr);

		if(itmax-2-it+1>=delt)
		{
			pre=0.0;
			pst=0.0;

			for(ii=0;ii<delt;ii++)
			{
				pre+=fxz[ii*ntp+ip];
				pst+=fxzr[ii*ntp+ip];
			}

			tht=pre/pst;
			ftheta=0.5*acosf(tht)*180.0/PI;
/*
			itheta=abs(ceilf(ftheta-0.5));
			if(itheta<thetan)
				image_ppcig[ipp+itheta]+=p_inv[ip]*p[ip];//fx*fxr+fz*fzr;//image_pp[ip];
*/
			for(ii=-2;ii<=2;ii++)
			{
				itheta=ceilf(ftheta+ii-0.5);//abs(ceilf(ftheta+ii-0.5));//

				betaf=1.0*ii*ii/(2.0*9.0);//2*sig^2

				if(itheta>=2&&itheta<thetan)
				{
					image_ppcig[ipp+itheta]+=p_inv[ip]*p[ip]*expf(-betaf);
				}
			}

		}
	}
	__syncthreads();
}

__global__ void laplace_kernel_image
(
 float *image, float *image_ppf, 
 int ntx, int ntz, int pml, int Lc, float *rc, float dx, float dz
 )

{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int pmlc=pml+Lc;

	int ip=ix*ntz+iz;
	float diff1=0.0;
	float diff2=0.0;

	if(iz>=pmlc&&iz<=ntz-pmlc-1&&ix>=pmlc&&ix<=ntx-pmlc-1)
	{
		diff1=(image[ip+1]-2.0*image[ip]+image[ip-1])/(dz*dz);
		diff2=(image[ip+1*ntz]-2.0*image[ip]+image[ip-1*ntz])/(dx*dx);	
	}

	image_ppf[ip]=diff1+diff2;

	__syncthreads();
}

/*==========================================================

  This subroutine is used for calculating wave field in 2D.

  ===========================================================*/

extern "C"
void fdtd_2d_GPU_backward(int ntx, int ntz, int ntp, int nx, int nz,
		int pml, int Lc, float *rc, float dx, float dz,
		float *rick, int itmax, float dt, int myid, int thetan, int delt,
		int is, struct Source ss[], struct MultiGPU plan[], int GPU_N, int rnmax,
		float *vp, float *rho, int *mx,
		float *k_x, float *k_x_half,
		float *k_z, float *k_z_half,
		float *a_x, float *a_x_half,
		float *a_z, float *a_z_half,
		float *b_x, float *b_x_half,
		float *b_z, float *b_z_half)
{
	int i,it,ip,tt;
	int ix,iz;
	int pmlc=pml+Lc;

	FILE *fp;
	char filename[40];

	// vectors for the devices

	size_t size_model=sizeof(float)*ntp;

	//    int iz,ix;

	Multistream plans[GPU_N];

	// allocate the memory for the device
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		cudaStreamCreate(&plans[i].stream);	
	}

	// =============================================================================

	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 dimGrid((ntx+dimBlock.x-1)/dimBlock.x,(ntz+dimBlock.y-1)/dimBlock.y);

	//-----------------------------------------------------------------------//
	//=======================================================================//
	//-----------------------------------------------------------------------//

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		sprintf(filename,"./output/wavefield%d_itmax%d.dat",myid,i);
		fp=fopen(filename,"rb");
		fread(&plan[i].vx[0],sizeof(float),ntp,fp);
		fread(&plan[i].vz[0],sizeof(float),ntp,fp);

		fread(&plan[i].p[0],sizeof(float),ntp,fp);
		fclose(fp);

		cudaMemcpyAsync(plan[i].d_vx_inv,plan[i].vx,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_vz_inv,plan[i].vz,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_p_inv,plan[i].p,size_model,cudaMemcpyHostToDevice,plans[i].stream);

		cudaStreamSynchronize(plans[i].stream);
	}

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		// Copy the vectors from the host to the device

		cudaMemcpyAsync(plan[i].d_seismogram_rms,plan[i].seismogram_rms,
				sizeof(float)*ss[is+i].r_n*itmax,cudaMemcpyHostToDevice,plans[i].stream);
/*
		cudaMemcpyAsync(plan[i].d_p_borders_up,plan[i].p_borders_up,
				sizeof(float)*Lc*nx*itmax,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_p_borders_bottom,plan[i].p_borders_bottom,
				sizeof(float)*Lc*nx*itmax,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_p_borders_left,plan[i].p_borders_left,
				sizeof(float)*Lc*(nz-2*Lc)*itmax,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_p_borders_right,plan[i].p_borders_right,
				sizeof(float)*Lc*(nz-2*Lc)*itmax,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_rick,rick,sizeof(float)*itmax,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_rc,rc,sizeof(float)*itmax,cudaMemcpyHostToDevice,plans[i].stream);
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
		cudaMemcpyAsync(plan[i].d_b_z_half,b_z_half,sizeof(float)*ntz,cudaMemcpyHostToDevice,plans[i].stream);*/
	}

	// Initialize the fields........................

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		cudaMemsetAsync(plan[i].d_image_pp,0.0,size_model,plans[i].stream);
		cudaMemsetAsync(plan[i].d_image_ppcig,0.0,size_model,plans[i].stream);
		cudaMemsetAsync(plan[i].d_image_sources,0.0,size_model,plans[i].stream);
		cudaMemsetAsync(plan[i].d_image_receivers,0.0,size_model,plans[i].stream);

		cudaMemsetAsync(plan[i].d_vx,0.0,size_model,plans[i].stream);
		cudaMemsetAsync(plan[i].d_vz,0.0,size_model,plans[i].stream);
		cudaMemsetAsync(plan[i].d_p,0.0,size_model,plans[i].stream);

		cudaMemsetAsync(plan[i].dp_dt,0.0,size_model,plans[i].stream);

		cudaMemsetAsync(plan[i].d_phi_vx_x,0.0,size_model,plans[i].stream);
		cudaMemsetAsync(plan[i].d_phi_vz_z,0.0,size_model,plans[i].stream);

		cudaMemsetAsync(plan[i].d_phi_p_x,0.0,size_model,plans[i].stream);
		cudaMemsetAsync(plan[i].d_phi_p_z,0.0,size_model,plans[i].stream);
	}

	//==============================================================================
	//  THIS SECTION IS USED TO CONSTRUCT THE FORWARD WAVEFIELDS...           
	//==============================================================================


	for(it=itmax-2;it>=0;it--)
	{
		for(i=0;i<GPU_N;i++)
		{
			cudaSetDevice(i);

			fdtd_2d_GPU_kernel_p_backward<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rick, plan[i].d_vp, plan[i].d_rho, plan[i].d_mx,
				 plan[i].d_vx_inv, plan[i].d_vz_inv, plan[i].d_p_inv,
				 ntp, ntx, ntz, pml, Lc, plan[i].d_rc, dx, dz, -dt,
				 ss[is+i].s_ix, ss[is+i].s_iz, it, plan[i].dp_dt
				);

			fdtd_2d_GPU_kernel_borders_backward<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_p_inv,
				 plan[i].d_p_borders_up, plan[i].d_p_borders_bottom,
				 plan[i].d_p_borders_left, plan[i].d_p_borders_right,
				 ntp, ntx, ntz, pml, Lc, plan[i].d_rc, it, itmax
				);

			fdtd_2d_GPU_kernel_vx_backward<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rho, plan[i].d_mx, 
				 plan[i].d_vx_inv, plan[i].d_p_inv,
				 ntp, ntx, ntz, pml, Lc, plan[i].d_rc, dx, dz, -dt
				);

			fdtd_2d_GPU_kernel_vz_backward<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rho, plan[i].d_mx,
				 plan[i].d_vz_inv, plan[i].d_p_inv, 
				 ntp, ntx, ntz, pml, Lc, plan[i].d_rc, dx, dz, -dt
				);

			///////////////////////////////////////////////////////////////////////

			fdtd_cpml_2d_GPU_kernel_vx_backward<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rho, plan[i].d_mx, plan[i].d_a_x_half, plan[i].d_a_z,
				 plan[i].d_b_x_half, plan[i].d_b_z,
				 plan[i].d_vx, plan[i].d_p,
				 plan[i].d_phi_p_x,
				 ntp, ntx, ntz, -dx, -dz, dt,
				 it, pml, Lc, plan[i].d_rc
				);

			fdtd_cpml_2d_GPU_kernel_vz_backward<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rho, plan[i].d_mx,
				 plan[i].d_a_x, plan[i].d_a_z_half,
				 plan[i].d_b_x, plan[i].d_b_z_half,
				 plan[i].d_vz, plan[i].d_p,
				 plan[i].d_phi_p_z,
				 ntp, ntx, ntz, -dx, -dz, dt,
				 it, pml, Lc, plan[i].d_rc
				);

			fdtd_cpml_2d_GPU_kernel_p_backward<<<dimGrid,dimBlock,0,plans[i].stream>>>
				( 
				 plan[i].d_vp, plan[i].d_rho, plan[i].d_mx,
				 plan[i].d_a_x, plan[i].d_a_z, plan[i].d_b_x, plan[i].d_b_z,
				 plan[i].d_vx, plan[i].d_vz, plan[i].d_p,
				 plan[i].d_phi_vx_x, plan[i].d_phi_vz_z,
				 ntp, ntx, ntz, pml, Lc, plan[i].d_rc,
				 plan[i].d_seismogram_rms, ss[is+i].r_ix, plan[i].d_r_iz, ss[is+i].r_n, -dx, -dz, dt, itmax, it
				);

			sum_image_GPU_kernel_image<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_vp, plan[i].d_rho, plan[i].d_mx,
				 plan[i].d_p_inv, plan[i].dp_dt, plan[i].d_vx_inv, plan[i].d_vz_inv, plan[i].d_p, plan[i].d_vx, plan[i].d_vz,
				 plan[i].d_image_pp, plan[i].d_image_sources, plan[i].d_image_receivers, ss[is+i].s_ix, ss[is+i].r_ix,
				 ntx, ntz, thetan, pml, Lc, plan[i].d_rc, dx, dz
				);
/*
			sum_image_GPU_kernel_imageCIG<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_vp, plan[i].d_rho, plan[i].d_mx, plan[i].d_fxz, plan[i].d_fxzr,
				 plan[i].d_p_inv, plan[i].dp_dt, plan[i].d_vx_inv, plan[i].d_vz_inv, plan[i].d_p, plan[i].d_vx, plan[i].d_vz,
				 plan[i].d_image_pp, plan[i].d_image_ppcig,
				 ntx, ntz, thetan, delt, pml, Lc, plan[i].d_rc, dx, dz, it, itmax
				);

			if(it%10==0)
			{
				cudaMemcpyAsync(plan[i].vx,plan[i].d_p_inv,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
				cudaMemcpyAsync(plan[i].vz,plan[i].d_p,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
				cudaStreamSynchronize(plans[i].stream);

				sprintf(filename,"./output/%dvx%d_inv.dat",it,i);     
				fp=fopen(filename,"wb");
				for(ix=pmlc;ix<ntx-pmlc;ix++)
				{
					for(iz=pmlc;iz<ntz-pmlc;iz++)
					{					
						fwrite(&plan[i].vx[ix*ntz+iz],sizeof(float),1,fp);
					}
				}
				fclose(fp);

				sprintf(filename,"./output/%dvx%d_bak.dat",it,i);     
				fp=fopen(filename,"wb");
				for(ix=pmlc;ix<ntx-pmlc;ix++)
				{
					for(iz=pmlc;iz<ntz-pmlc;iz++)
					{					
						fwrite(&plan[i].vz[ix*ntz+iz],sizeof(float),1,fp);
					}
				}
				fclose(fp);
			}
*/
		}//end GPU_N
	}//end it

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		
		cudaMemcpyAsync(plan[i].image_pp,plan[i].d_image_pp,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
		cudaMemcpyAsync(plan[i].image_ppcig,plan[i].d_image_ppcig,sizeof(float)*ntp*thetan,cudaMemcpyDeviceToHost,plans[i].stream);

		cudaMemcpyAsync(plan[i].image_sources,plan[i].d_image_sources,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
		cudaMemcpyAsync(plan[i].image_receivers,plan[i].d_image_receivers,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);

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

	// ==========================================================
	// allocate the memory of Vx,Vy,Vz,Sigmaxx,Sigmayy,...
/*
	cudaMallocHost((void **)&rick,sizeof(float)*itmax);  
	cudaMallocHost((void **)&rc,sizeof(float)*Lc);  

	cudaMallocHost((void **)&lambda,sizeof(float)*ntp); 
	cudaMallocHost((void **)&mu,sizeof(float)*ntp); 
	cudaMallocHost((void **)&rho,sizeof(float)*ntp); 
	cudaMallocHost((void **)&lambda_plus_two_mu,sizeof(float)*ntp); 

	cudaMallocHost((void **)&a_x,sizeof(float)*ntx);
	cudaMallocHost((void **)&a_x_half,sizeof(float)*ntx);
	cudaMallocHost((void **)&a_z,sizeof(float)*ntz);
	cudaMallocHost((void **)&a_z_half,sizeof(float)*ntz);

	cudaMallocHost((void **)&b_x,sizeof(float)*ntx);
	cudaMallocHost((void **)&b_x_half,sizeof(float)*ntx);
	cudaMallocHost((void **)&b_z,sizeof(float)*ntz);
	cudaMallocHost((void **)&b_z_half,sizeof(float)*ntz);

	cudaMallocHost((void **)&vx,sizeof(float)*ntp); 
	cudaMallocHost((void **)&vz,sizeof(float)*ntp); 
	cudaMallocHost((void **)&sigmaxx,sizeof(float)*ntp);
	cudaMallocHost((void **)&sigmazz,sizeof(float)*ntp);
	cudaMallocHost((void **)&sigmaxz,sizeof(float)*ntp);
*/

	// allocate the memory for the device
	// allocate the memory for the device
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		// allocate the memory of vx,vy,vz,sigmaxx,sigmayy,...
		plan[i].vx=(float*)malloc(sizeof(float)*ntp); 
		plan[i].vz=(float*)malloc(sizeof(float)*ntp); 
		plan[i].p=(float*)malloc(sizeof(float)*ntp);

		plan[i].phi_vx_x      = (float*)malloc(sizeof(float)*ntp);
		plan[i].phi_vz_z      = (float*)malloc(sizeof(float)*ntp);

		plan[i].phi_p_x=(float*)malloc(sizeof(float)*ntp);
		plan[i].phi_p_z=(float*)malloc(sizeof(float)*ntp);

		///device//////////
		///device//////////
		///device//////////

		cudaMalloc((void**)&plan[i].d_fxz,delt*size_model);
		cudaMalloc((void**)&plan[i].d_fxzr,delt*size_model);

		cudaMalloc((void**)&plan[i].d_seismogram,sizeof(float)*itmax*rnmax);
		cudaMalloc((void**)&plan[i].d_seismogram_rms,sizeof(float)*itmax*(rnmax));

		cudaMalloc((void**)&plan[i].d_r_iz,sizeof(int)*rnmax);

		cudaMalloc((void**)&plan[i].d_rick,sizeof(float)*itmax);        // ricker wave 
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

		cudaMalloc((void**)&plan[i].d_image_pp,size_model);
		cudaMalloc((void**)&plan[i].d_image_ppcig,size_model*thetan);

		cudaMalloc((void**)&plan[i].d_image_sources,size_model);
		cudaMalloc((void**)&plan[i].d_image_receivers,size_model);

		cudaMalloc((void**)&plan[i].d_vx,size_model);
		cudaMalloc((void**)&plan[i].d_vz,size_model);
		cudaMalloc((void**)&plan[i].d_p,size_model);

		cudaMalloc((void**)&plan[i].dp_dt,size_model);

		cudaMalloc((void**)&plan[i].d_vx_inv,size_model);
		cudaMalloc((void**)&plan[i].d_vz_inv,size_model);
		cudaMalloc((void**)&plan[i].d_p_inv,size_model);

		cudaMalloc((void**)&plan[i].d_phi_vx_x,size_model);
		cudaMalloc((void**)&plan[i].d_phi_vz_z,size_model);

		cudaMalloc((void**)&plan[i].d_phi_p_x,size_model);
		cudaMalloc((void**)&plan[i].d_phi_p_z,size_model);


		cudaMalloc((void**)&plan[i].d_p_borders_up,sizeof(float)*Lc*itmax*nx);
		cudaMalloc((void**)&plan[i].d_p_borders_bottom,sizeof(float)*Lc*itmax*nx);
		cudaMalloc((void**)&plan[i].d_p_borders_left,sizeof(float)*Lc*itmax*(nz-2*Lc));
		cudaMalloc((void**)&plan[i].d_p_borders_right,sizeof(float)*Lc*itmax*(nz-2*Lc));

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
/*
	cudaFreeHost(rick); 
	cudaFreeHost(rc); 
	
	//free the memory of lambda
	cudaFreeHost(lambda); 
	cudaFreeHost(mu); 
	cudaFreeHost(rho); 
	cudaFreeHost(lambda_plus_two_mu); 

	cudaFreeHost(a_x);
	cudaFreeHost(a_x_half);
	cudaFreeHost(a_z);
	cudaFreeHost(a_z_half);

	cudaFreeHost(b_x);
	cudaFreeHost(b_x_half);
	cudaFreeHost(b_z);
	cudaFreeHost(b_z_half);

	cudaFreeHost(vx);
	cudaFreeHost(vz);
	cudaFreeHost(sigmaxx);
	cudaFreeHost(sigmazz);
	cudaFreeHost(sigmaxz);
*/	 
	//free the memory of DEVICE
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		//free the memory of Vx,Vy,Vz,Sigmaxx,Sigmayy,Sigmazz...  
		free(plan[i].vx);
		free(plan[i].vz);
		free(plan[i].p);

		//free the memory of Phi_vx_x....  
		free(plan[i].phi_vx_x);
		free(plan[i].phi_vz_z);

		//free the memory of Phi_vx_x....  
		free(plan[i].phi_p_x);
		free(plan[i].phi_p_z);
		////device////
		////device////
		////device////

		cudaFree(plan[i].d_fxz);
		cudaFree(plan[i].d_fxzr);

		cudaFree(plan[i].d_seismogram);
		cudaFree(plan[i].d_seismogram_rms);

		cudaFree(plan[i].d_r_iz);

		cudaFree(plan[i].d_rick);
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

		cudaFree(plan[i].d_image_pp);
		cudaFree(plan[i].d_image_ppcig);

		cudaFree(plan[i].d_image_sources);
		cudaFree(plan[i].d_image_receivers);

		cudaFree(plan[i].d_vx);
		cudaFree(plan[i].d_vz);
		cudaFree(plan[i].d_p);

		cudaFree(plan[i].dp_dt);

		cudaFree(plan[i].d_vx_inv);
		cudaFree(plan[i].d_vz_inv);
		cudaFree(plan[i].d_p_inv);

		cudaFree(plan[i].d_phi_vx_x);
		cudaFree(plan[i].d_phi_vz_z);
		cudaFree(plan[i].d_phi_p_x);
		cudaFree(plan[i].d_phi_p_z);

		cudaFree(plan[i].d_p_borders_up);
		cudaFree(plan[i].d_p_borders_bottom);
		cudaFree(plan[i].d_p_borders_left);
		cudaFree(plan[i].d_p_borders_right);
	}
}

extern "C"
void getdevice(int *GPU_N)
{
	
	cudaGetDeviceCount(GPU_N);	
//	GPU_N=6;//4;//2;//
}

