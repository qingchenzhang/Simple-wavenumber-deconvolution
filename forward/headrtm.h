#include"cufft.h"

#define BLOCK_SIZE 8
#define OUTPUT_SNAP 0
#define PI 3.1415926

void ricker_phase_shift(int itmax, float *rick);

extern "C"
void getdevice(int *GPU_N);

extern "C"
struct Source
{
	int s_iz,s_ix,r_ix,r_n;
	int *r_iz;
};

extern "C"
struct MultiGPU
{
	float *coef;

	float *vx,*vz;
	cufftComplex *p,*htp;
	cufftComplex *spu,*spd;
	cufftComplex *rpu,*rpd;

	float *vx_inv,*vz_inv;
	float *p_inv;  

	float *phi_vx_x,*phi_vz_z;
	float *phi_p_x,*phi_p_z;

	float *seismogram_obs;
	float *seismogram_up;
	float *seismogram_down;
	float *seismogram_dir;
	float *seismogram_dirup;
	float *seismogram_dirdown;
	float *seismogram_syn;
	float *seismogram_rms;
	float *seismogram_rmsup;
	float *seismogram_rmsdown;
	float *seismogram_hrms;

	float *p_borders_up,   *p_borders_bottom;
	float *p_borders_left, *p_borders_right;

	float *image_pp;
	float *image_ppdu,*image_ppud,*image_ppcig;
	float *image_sources,*image_receivers;

	// vectors for the devices
	float *d_coef;
	int *d_r_iz;
	
	float *d_rick;
	float *d_hrick;
	float *d_rc;
	float *d_asr;

	float *d_vp, *d_rho;
	int *d_mx;

	float *d_a_x, *d_a_x_half;
	float *d_a_z, *d_a_z_half;
	float *d_b_x, *d_b_x_half;
	float *d_b_z, *d_b_z_half;

	float *d_vx,*d_vz;
	//float *d_p;
	cufftComplex *d_p;

	float *d_htvx,*d_htvz;
	//float *d_htp;
	cufftComplex *d_htp;

	///////////////////////
	cufftComplex *d_hp;
	cufftComplex *d_temp;
	cufftComplex *d_hhtp;
	///////////////////////
	////////////////////////////////////////
	////////// Seperated Wavefield//////////
	cufftComplex *d_spu;
	cufftComplex *d_spd;
	cufftComplex *d_rpu;
	cufftComplex *d_rpd;

	float *d_vx_inv,*d_vz_inv;
	//float *d_p_inv;  
	cufftComplex *d_p_inv;  

	float *d_htvx_inv,*d_htvz_inv;
	//float *d_htp_inv;  
	cufftComplex *d_htp_inv;  

	float *d_phi_vx_x,*d_phi_vz_z;
	float *d_phi_p_x,*d_phi_p_z;

	float *d_phi_htvx_x,*d_phi_htvz_z;
	float *d_phi_htp_x,*d_phi_htp_z;

	float *d_seismogram;
	float *d_seismogram_up;
	float *d_seismogram_down;
	float *d_seismogram_rms;
	float *d_seismogram_hrms;

	float *d_p_borders_up,*d_p_borders_bottom;
	float *d_p_borders_left,*d_p_borders_right;

	float *d_htp_borders_up,*d_htp_borders_bottom;
	float *d_htp_borders_left,*d_htp_borders_right;

	float *dp_dt;
	float *d_image_pp;
	float *d_image_ppdu,*d_image_ppud,*d_image_ppcig;
	float *d_image_sources,*d_image_receivers;

	float *d_fxz,*d_fxzr;
	//
	//////////////////////////
	cufftHandle forward_z,backward_z;
	
};

void get_acc_model(float *vp, float *rho, int ntp, int ntx, int ntz, int pml, char *v_file);

void ini_model_minea(float *vp, float *rho, int ntp, int ntz, int ntx, int pml, int flag);

void ini_model_mine(float *vp, float *vp_n, int ntp, int ntz, int ntx, int pml, int window);

void get_ini_model(float *vp, float *rho, int ntp, int ntx, int ntz, int pml);
	
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
		int pml, float dx, float dz, float f0, float t0, float dt, float vp_max);

void maximum_vector(float *vector, int n, float *maximum_value);

void get_lame_constants( 
		float *vp_two, float *vp, 
		float * rho, int ntp);

void ini_step(float *dn, int np, float *un0, float max);

void ricker_wave(float *rick, int itmax, float f0, float t0, float dt, int flag);

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
		float *c, int inv_flag);

//////////////////////////////////////////////////////////
extern "C"
void fdtd_2d_GPU_backward(int ntx, int ntz, int ntp, int nx, int nz,
		int pml, int Lc, float *rc, float dx, float dz,
		float *rick, float *hrick, int itmax, float dt, int myid, int thetan, int delt,
		int is, struct Source ss[], struct MultiGPU plan[], int GPU_N, int rnmax, 
		float *vp, float *rho, int *mx,
		float *k_x, float *k_x_half,
		float *k_z, float *k_z_half,
		float *a_x, float *a_x_half,
		float *a_z, float *a_z_half,
		float *b_x, float *b_x_half,
		float *b_z, float *b_z_half);

void update_model(float *vp, float *vp_n,
		float *dn_vp, float *un_vp,
		int ntp, int ntz, int ntx, int pml, float vpmin, float vpmax);

void Preprocess(int nz, int nx, float dx, float dz, float *P);

void cal_xishu(int Lx,float *rx);

extern "C"
void congpu_fre(float *seismogram_syn, float *seismogram_obs, float *seismogram_rms, float *Misfit, int i, 
		float *ref_obs, float *ref_syn, int itmax, int nx, int s_ix, int *r_ix, int pml);

extern "C"
void laplacegpu_filter(float *image, int i, int nxx,
		int nzz, float dx, float dz, int pml);

extern "C"
void rmsgpu_fre(float *seismogram_rms,int tmax,int nx,int i,float dt);

extern "C"
void variables_malloc(int ntx, int ntz, int ntp, int nx, int nz,
		int pml, int Lc, float dx, float dz, int itmax, int thetan, int delt,
		struct MultiGPU plan[], int GPU_N, int rnmax
		);

extern "C"
void variables_free(int ntx, int ntz, int ntp, int nx, int nz,
		int pml, int Lc, float dx, float dz, int itmax,
		struct MultiGPU plan[], int GPU_N, int rnmax
		);

//void input_parameters(int *nx,int *nz,int *pml,int *Lc,float *dx,float *dz,float *rectime,float *dt,float *f0, 
//		int *ns,  float *sx0,float *shotdx,float *shotdep,
//		int *r_n, float *rx0,float *recdx, float *recdep,char *v_file,char *data_file,char *srn_file,char *sx_file,char *srx_file,char *mx_file
//		);
//void input_parameters(int *nx,int *nz,int *pml,int *Lc,float *dx,float *dz,float *rectime,float *dt,float *f0, 
void input_parameters(int *nx,int *nz,int *pml,int *Lc,float *dx,float *dz,int *itmax,float *dt,float *f0, 
		int *ns,float *sx0,float *shotdx,float *shotdep,
		float *recdep,float *offsetx,int *scatter_flag,int *swin,int *window
		);

extern "C"
void hilbert_transform1d(float *rick, float *rick_hilbert, int itmax);

extern "C"
void hilbert_transform2d(float *data, float *data_hilbert, int i, 
		int nz, int nx);
