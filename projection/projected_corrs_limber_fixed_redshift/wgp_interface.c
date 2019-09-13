/*An adapted version of the C(l) to xi_+/- code
Uses function tpstat_via_hankel from Martin Kilbinger's nicaea
*/
#include "cosmosis/datablock/c_datablock.h"
#include "cosmosis/datablock/section_names.h"
#include "maths.h"

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>

#define PI 3.14159 

const char * shear_xi = SHEAR_XI_SECTION;
const char * shear_cl = SHEAR_CL_SECTION;
const char * wl_nz = WL_NUMBER_DENSITY_SECTION;

typedef enum {gp=0, pp=1, mp=2, gg=3, mm=4, bb=5} corr_type_t;

typedef struct wgp_config {
	char * input_section;
	char * output_section;
	double redshift;
	corr_type_t corr_type;

} wgp_config;


void * setup(c_datablock * options)
{
	wgp_config * config = malloc(sizeof(wgp_config));
	int corr_type;
	double redshift;
	int status = 0;
	bool auto_corr;

	char *output_string=malloc(100*sizeof(char));
	char *input_section_name=malloc(100*sizeof(char));

	status |= c_datablock_get_int_default(options, OPTION_SECTION, "corr_type", 0, &corr_type);
	status |= c_datablock_get_double_default(options, OPTION_SECTION, "redshift", 0, &redshift);

	if (corr_type==gp){
	  sprintf(input_section_name,"galaxy_intrinsic_power_%2.3f",redshift);	
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "input_section_name", input_section_name, &(config->input_section));
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "output_section_name", "galaxy_intrinsic_w", &(config->output_section));

	}
	else if (corr_type==pp){
	  sprintf(input_section_name,"intrinsic_power_%2.3f",redshift);	
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "input_section_name", input_section_name, &(config->input_section));
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "output_section_name", "intrinsic_w", &(config->output_section));
	}
	else if (corr_type==bb){
	  sprintf(input_section_name,"intrinsic_power_bb_%2.3f",redshift);	
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "input_section_name", input_section_name, &(config->input_section));
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "output_section_name", "intrinsic_w_bb", &(config->output_section));
	}
	else if (corr_type==gg){
	  sprintf(input_section_name,"galaxy_power_%2.3f",redshift);	
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "input_section_name", input_section_name, &(config->input_section));
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "output_section_name", "galaxy_w", &(config->output_section));
	}
	else if (corr_type==mp){
	  sprintf(input_section_name,"matter_intrinsic_power_%2.3f",redshift);	
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "input_section_name", input_section_name, &(config->input_section));
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "output_section_name", "matter_intrinsic_w", &(config->output_section));
	}
	else if (corr_type==mm){
	  sprintf(input_section_name,"matter_power_%2.3f",redshift);	
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "input_section_name", input_section_name, &(config->input_section));
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "output_section_name", "matter_w", &(config->output_section));
	}
	else{
	  fprintf(stderr, "Unknown corr_type");
	  status = 1;
	}

	config->corr_type = (corr_type_t)corr_type;
	config->redshift = (double)redshift;

	if (status){
	  fprintf(stderr, "Please specify input_section_name, output_section_name, and corr_type=0,1,2,3 or 4.\n");
	  exit(status);
	}

	free(output_string);
	return config;
}

typedef struct p_projected_data{
	interTable * pk_table;
} p_projected_data;

static double P_projected_loglog(void *info, double k, int bin_i, int bin_j, error ** err)
{
	p_projected_data * data = (p_projected_data*) info;
	interTable * pk_table = data->pk_table;
	double pk;
	pk =  exp(interpol_wr(pk_table, log(k), err));
	return pk;
}

static double P_projected_logl(void *info, double k, int bin_i, int bin_j, error ** err)
{
	p_projected_data * data = (p_projected_data*) info;
	interTable * pk_table = data->pk_table;
	//FILE * f = data->f;
	double pk;
	// if (ell<20.0) cl = 0.0;
	// else if (ell>3000.0) cl = 0.0;
	// else
	pk =  interpol_wr(pk_table, log(k), err);
	//fprintf(f,"%le  %le\n", ell, cl);
	return pk;
}


int execute(c_datablock * block, void * config_in)
{
	DATABLOCK_STATUS status=0;
	int count=2;
	double ** W = malloc(count*sizeof(double*));
	for (int i=0; i<count; i++) W[i] = malloc(sizeof(double)*N_thetaH);

	wgp_config * config = (wgp_config*) config_in;

	//load k array
	double * kh;

	int n_k;
	status |= c_datablock_get_double_array_1d(block, config->input_section, "k_h", &kh, &n_k);
	if (status) {
		fprintf(stderr, "Could not load k in Pk -> xi\n");
		return status;
	}
	double log_k_min = log(kh[0]);
	double log_k_max = log(kh[n_k-1]);
	double dlog_k=(log_k_max-log_k_min)/(n_k-1);
	error * err = NULL;
	interTable* pk_table; 

	char name_in[64],name_w1[64],name_w2[64];

	double log_rp_min = log(rp_min);
	double log_rp_max = log(rp_max);

	// Load k and P, and compute w
	bool found_any = false;
	double * Pk;
	snprintf(name_in, 64, "p_k");
	found_any=true;
	status |= c_datablock_get_double_array_1d(block, config->input_section, name_in, &Pk, &n_k);
	if (status) {
		fprintf(stderr, "Could not load P(k)");
		return status;
	}

	double coeff;

	// Choose the type of Hankel transform
	char *name=malloc(100*sizeof(char));
	tpstat_t tpstat;
	switch(config->corr_type) {
		case gp:
		    tpstat = tp_gt;
		    sprintf(name, "w_rp_limber_%3.3f", config->redshift);
		    snprintf(name_w1, 64, name);
		    coeff = -1.0 ;///2.0/PI;
		    break;
		case pp:
		    tpstat = tp_xipm;
		    sprintf(name, "w_rp_limber_%3.3f", config->redshift);
		    snprintf(name_w1, 64, name);
		    //snprintf(name_w2, 64, "w_rp_j4_limber_%3.3f", wgp_config->redshift);
		    coeff = 1.0; //2.0 * 2.0 * PI; ///4.0/PI;
		    break;
		case bb:
		    tpstat = tp_xipm;
		    sprintf(name, "w_rp_limber_%3.3f", config->redshift);
		    snprintf(name_w1, 64, name);
		    //snprintf(name_w2, 64, "w_rp_j4_limber_%3.3f", wgp_config->redshift);
		    coeff = 1.0; //2.0 * 2.0 * PI; ///4.0/PI;
		    break;
		case gg:
		    tpstat = tp_w;
		    sprintf(name, "w_rp_limber_%3.3f", config->redshift);
		    snprintf(name_w1, 64, name);
		    coeff = 1.0 ; //1.0/2.0/PI;
		    break;
		case mm:
		    tpstat = tp_w;
		    sprintf(name, "w_rp_limber_%3.3f", config->redshift);
		    snprintf(name_w1, 64, name);
		    coeff = 1.0 ; //1.0/2.0/PI;
		    break;
		case mp:
		    tpstat = tp_gt;
		    sprintf(name, "w_rp_limber_%3.3f", config->redshift);
		    snprintf(name_w1, 64, name);
		    coeff = -1.0;
		    break;
		default:
		    printf("corr_type: %d\n", config->corr_type);
		    printf("ERROR: Invalid corr_type %d in wgp_interface\n",config->corr_type);
		    return 10;
	}

    //need to check for zero or negative values...if there are some, can't do loglog interpolation
    int neg_vals=0;
    for (int i=0; i<n_k; i++){
    	if (Pk[i]<=0) {
    		neg_vals += 1;
    	}
	}

	//also check for zeros, and replace with v small number if all other C_ells are all +ve or -ve
	for (int i=0; i<n_k; i++){
		if (fabs(Pk[i])<=1.e-30) {
			if (neg_vals==n_k){
				Pk[i]=-1.e-30;
			}
			else if (neg_vals==0) {
				Pk[i]=1.e-30;
			}
		}
	}	

	if (neg_vals == 0) {
		pk_table = init_interTable(n_k, log_k_min, log_k_max, dlog_k, 1.0, -3.0, &err);
		for (int i=0; i<n_k; i++){
			pk_table->table[i] = log(Pk[i]);
		}
	}

	else if (neg_vals == n_k){
	    //In this case all the P(k) elements are negative, so interpolate in log(-P)
	    //just remember to flip sign again at the end
		pk_table = init_interTable(n_k, log_k_min, log_k_max, dlog_k, 1.0, -3.0, &err);
		for (int i=0; i<n_k; i++){
			pk_table->table[i] = log(-Pk[i]);
		}
	}
	else {
		static int warned=0;
		if (warned==0){
			printf("Negative values in P(k). No interpolation in log(P(k)),\n");
			printf("and no power law extrapolation. So make sure range of input k\n");
			printf("is sufficient for this not to matter. \n");
			printf("This warning will only appear once per process. \n");
			warned=1;
		}
		pk_table = init_interTable(n_k, log_k_min, log_k_max, dlog_k, 0., 0., &err);
		for (int i=0; i<n_k; i++){
			pk_table->table[i] = Pk[i];
		}
	}

	p_projected_data d;
	d.pk_table = pk_table;
	//d.f = f;
	if (neg_vals == 0) {
		tpstat_via_hankel(&d, W, &log_rp_min, &log_rp_max, tpstat, &P_projected_loglog, 1, 1, &err);
	}
	else if (neg_vals == n_k) {
		tpstat_via_hankel(&d, W, &log_rp_min, &log_rp_max, tpstat, &P_projected_loglog, 1, 1, &err);
		double w1_orig,w2_orig;
		for (int i=0; i<N_thetaH; i++){
			w1_orig = W[0][i];
			W[0][i] =-1*w1_orig;
			if (config->corr_type == pp) {
				w2_orig = W[1][i];
				W[1][i] =-1*w2_orig;
			}
		}
	}
	else {
		tpstat_via_hankel(&d, W, &log_rp_min, &log_rp_max, tpstat, &P_projected_logl, 1, 1, &err);
	}

	// Also include the prefactor terms
	for (int i=0; i<N_thetaH; i++){
		W[0][i] = coeff * W[0][i];
		if (config->corr_type == pp) {
			W[1][i] = coeff * W[1][i];
			W[0][i]+= W[1][i] ;
		}
		if (config->corr_type == bb) {
			W[1][i] = coeff * W[1][i];
			W[0][i]-= W[1][i] ;
		}
	}
				

	//Now save to block
	c_datablock_put_double_array_1d(block, config->output_section, name_w1, W[0], N_thetaH);

	free(Pk);
	del_interTable(&d.pk_table);

	if (!found_any){
		fprintf(stderr, "WARNING: I COULD NOT FIND ANY SPECTRA OF THE FORM \n");
		fprintf(stderr, "xiplus_i_j, ximinus_i_j, wmatter_i_j, or tanshear_i_j.\n");
		fprintf(stderr, "THIS IS ALMOST CERTAINLY AN ERROR.\n");
		status = 1;
	}
	// Save theta values...check these are being calculated correctly at some point
	double dlog_rp= (log_rp_max-log_rp_min)/((double)N_thetaH-1.0);
	double logrp_center = 0.5*(log_rp_max+log_rp_min);
	int nc = N_thetaH/2+1;
	double rp_vals[N_thetaH];

	for (int i; i<N_thetaH; i++){
		rp_vals[i] = exp(log_rp_min+i*dlog_rp);
	}

	c_datablock_put_double_array_1d(block, config->output_section, "r_p", rp_vals, N_thetaH);

	//Clean up
	for (int i=0; i<count; i++) free(W[i]);
	free(W);
	free(kh);

	return status;
}

int cleanup(void * config_in)
{
	// Free the memory that we allocated in the setup
	wgp_config * config = (wgp_config*) config_in;
	free(config);
}
