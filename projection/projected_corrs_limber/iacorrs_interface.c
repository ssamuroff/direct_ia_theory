/*CosmoSIS interface file for going from shear C(l) to xi+/-
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

typedef enum {gp=0, pp=1, mp=2, gg=3, mm=4, gp1h=5} corr_type_t;

typedef struct ia_config {
	char * input_section;
	char * sample_a;
	char * sample_b;
	char * output_section;
	corr_type_t corr_type;
} ia_config;


void * setup(c_datablock * options)
{
	ia_config * config = malloc(sizeof(ia_config));
	int corr_type;
	int status = 0;
	bool auto_corr;

	char *output_string=malloc(100*sizeof(char));
	char *input_section_name=malloc(100*sizeof(char));


	status |= c_datablock_get_int_default(options, OPTION_SECTION, "corr_type", 0, &corr_type);
	status |= c_datablock_get_string_default(options, OPTION_SECTION, "sample_a", "sample_a", &(config->sample_a));
	status |= c_datablock_get_string_default(options, OPTION_SECTION, "sample_b", "sample_b", &(config->sample_b));

	if (corr_type==gp){
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "input_section_name", "galaxy_intrinsic_power_projected", &(config->input_section));
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "output_section_name", "galaxy_intrinsic_w", &(config->output_section));

	}
	else if (corr_type==gp1h){
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "input_section_name", "galaxy_intrinsic_power_1h_projected", &(config->input_section));
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "output_section_name", "galaxy_intrinsic_w_1h", &(config->output_section));

	}
	else if (corr_type==pp){
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "input_section_name", "intrinsic_power_projected", &(config->input_section));
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "output_section_name", "intrinsic_w", &(config->output_section));
	}
	else if (corr_type==gg){
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "input_section_name", "galaxy_power_projected", &(config->input_section));
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "output_section_name", "galaxy_w", &(config->output_section));
	}
	else if (corr_type==mp){
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "input_section_name", "matter_intrinsic_power_projected", &(config->input_section));
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "output_section_name", "matter_intrinsic_w", &(config->output_section));
	}
	else if (corr_type==mm){
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "input_section_name", "matter_power_projected", &(config->input_section));
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "output_section_name", "matter_w", &(config->output_section));
	}
	else{
	  fprintf(stderr, "Unknown corr_type");
	  status = 1;
	}

	config->corr_type = (corr_type_t)corr_type;

	if (status){
	  fprintf(stderr, "Please specify input_section_name, output_section_name, and corr_type=0,1,2,3,4 or 5.\n");
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
	int num_z_bin_A;
	int num_z_bin_B;
	int count=2;
	double ** W = malloc(count*sizeof(double*));
	for (int i=0; i<count; i++) W[i] = malloc(sizeof(double)*N_thetaH);

	ia_config * config = (ia_config*) config_in;

    // Load the number of redshift bins
	if (c_datablock_has_value(block, config->input_section, "nbin_a")){
		status |= c_datablock_get_int(block, config->input_section, "nbin_a", &num_z_bin_A);
		status |= c_datablock_get_int(block, config->input_section, "nbin_b", &num_z_bin_B);
	}
	else{
		status |= c_datablock_get_int(block, config->input_section, "nbin", &num_z_bin_A);
		num_z_bin_B = num_z_bin_A;
	}

	if (status) {
		printf("Exit status: %d \n", status);
		fprintf(stderr, "Could not load bin information\n");
		return status;
	}

	//load k array
	double * kh;
	int n_k;
	printf("Converting power spectrum to w(r_p): \n");
	printf(config->input_section);
	printf("...\n");
	status |= c_datablock_get_double_array_1d(block, "matter_power_nl", "k_h", &kh, &n_k);
	if (status) {
		printf("Exit status: %d \n", status);
		fprintf(stderr, "Could not load k\n");
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

	int j_bin_start;
	bool found_any = false;
	for (int i_bin=1; i_bin<=num_z_bin_A; i_bin++) {
		for (int j_bin=1; j_bin<=num_z_bin_B; j_bin++) {
		    // Load k and P, and compute w
			double * Pk;
			snprintf(name_in, 64, "p_k_%d_%d_%s_%s",i_bin,j_bin,config->sample_a,config->sample_b);
			if (!c_datablock_has_value(block, config->input_section, name_in)) continue;
			found_any=true;
			status |= c_datablock_get_double_array_1d(block, config->input_section, name_in, &Pk, &n_k);
			if (status){
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
				    sprintf(name, "w_rp_%d_%d_%s_%s", i_bin, j_bin, config->sample_a, config->sample_b);
				    snprintf(name_w1, 64, name);
				    coeff = -1.0 ; // * sqrt(2) ;///2.0/PI;
				    break;
				case gp1h:
				    tpstat = tp_w;
				    sprintf(name, "w_rp_1h_%d_%d_%s_%s", i_bin, j_bin, config->sample_a, config->sample_b);
				    snprintf(name_w1, 64, name);
				    coeff = -1.0 ; // * sqrt(2) ;///2.0/PI;
				    break;
				case pp:
				    tpstat = tp_xipm;
				    sprintf(name, "w_rp_%d_%d_%s_%s", i_bin, j_bin, config->sample_a, config->sample_b);
				    snprintf(name_w1, 64, name);
				    coeff = 1.0 ; //2.0 * 2.0 * PI; ///4.0/PI;
				    break;
				case gg:
				    tpstat = tp_w;
				    sprintf(name, "w_rp_%d_%d_%s_%s", i_bin, j_bin, config->sample_a, config->sample_b);
				    snprintf(name_w1, 64, name);
				    coeff = 1.0 ; //1.0/2.0/PI;
				    break;
				case mm:
				    tpstat = tp_w;
				    sprintf(name, "w_rp_%d_%d_%s_%s", i_bin, j_bin, config->sample_a, config->sample_b);
				    snprintf(name_w1, 64, name);
				    coeff = 1.0 ; //1.0/2.0/PI;
				    break;
				case mp:
				    tpstat = tp_gt;
				    sprintf(name, "w_rp_%d_%d_%s_%s", i_bin, j_bin, config->sample_a, config->sample_b);
				    snprintf(name_w1, 64, name);
				    coeff = -1.0 ; // * sqrt(2);
				    break;
				default:
				    printf("corr_type: %d\n", config->corr_type);
				    printf("ERROR: Invalid corr_type %d in iacorrs_interface\n",config->corr_type);
				    return 10;
			}

			//need to check for zero or negative values...if there are some, can't do loglog interpolation
			int neg_vals=0;
			for (int i=0; i<n_k; i++){
				if (Pk[i]<=0) {
					neg_vals += 1;
				}
			}

			//also check for zeros, and replace with very small number if all other C_ells are all +ve or -ve
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
				for (int i=0; i<n_k; i++) pk_table->table[i] = log(Pk[i]);
			}

		    else if (neg_vals == n_k){
		        //In this case all the P(k) elements are negative, so interpolate in log(-P)
		        //just remember to flip sign again at the end
		        pk_table = init_interTable(n_k, log_k_min, log_k_max, dlog_k, 1.0, -3.0, &err);
		        for (int i=0; i<n_k; i++) pk_table->table[i] = log(-Pk[i]);
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

		    	for (int i=0; i<n_k; i++) pk_table->table[i] = Pk[i];
		    }

		    p_projected_data d;
		    d.pk_table = pk_table;

		    if (neg_vals == 0) {
		    	tpstat_via_hankel(&d, W, &log_rp_min, &log_rp_max, tpstat, &P_projected_loglog, 1, 1, &err);
		    }
		    else if (neg_vals == n_k) {
		    	tpstat_via_hankel(&d, W, &log_rp_min, &log_rp_max, tpstat, &P_projected_loglog, 1, 1, &err);
		    	double w1_orig,w2_orig;
		    	for (int i=0; i<N_thetaH; i++){
		    		w1_orig = W[0][i];
		    		W[0][i] =-1*w1_orig;
		    		if (config->corr_type == pp){
		    			w2_orig = W[1][i];
		    			W[1][i] =-1*w2_orig;
		    		}
		    	}
		    }

		    else tpstat_via_hankel(&d, W, &log_rp_min, &log_rp_max, tpstat, &P_projected_logl, 1, 1, &err);

		    // Also include the prefactor terms
		    for (int i=0; i<N_thetaH; i++){
		    	W[0][i] = coeff * W[0][i];
		    	if (config->corr_type == pp) {
		    		W[1][i] = coeff * W[1][i];
		    		W[0][i]+= W[1][i] ;
		    	}
		    }

		    //Now save to block
		    c_datablock_put_int(block, config->output_section, "nbin_A", num_z_bin_A);
		    c_datablock_put_int(block, config->output_section, "nbin_B", num_z_bin_B);
		    c_datablock_put_double_array_1d(block, config->output_section, name_w1, W[0], N_thetaH);
		   // printf("Saving %d %d \n",i_bin,j_bin);
		    free(Pk);
		    del_interTable(&d.pk_table);
		}
	}

	printf("Done all.");

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
	ia_config * config = (ia_config*) config_in;
	free(config);
}
