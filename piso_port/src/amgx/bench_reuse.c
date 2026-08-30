/* Does the AMG hierarchy survive a coefficient change?
 *
 * Our pressure matrix keeps its structure every step and its VALUES move by ~1e-3, because
 * Gamma = J/rowsum(A) drifts as the flow evolves. Three strategies, same six matrices:
 *   A  full setup each step                    -- the naive cost
 *   B  replace_coefficients + solver_resetup   -- AmgX's intended path
 *   C  replace_coefficients, NO resetup        -- reuse the hierarchy outright
 * C is fastest if it converges; the question is whether it does.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <amgx_c.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
                         return t.tv_sec + 1e-9*t.tv_nsec; }
static void cb(const char *m,int l){ (void)m;(void)l; }   /* silence AmgX chatter */

int main(int argc, char **argv){
    if(argc<3){ printf("usage: %s seq.bin config.json\n", argv[0]); return 1; }
    FILE *f=fopen(argv[1],"rb"); if(!f){printf("no file\n");return 1;}
    int h[3]; size_t r=fread(h,sizeof(int),3,f); (void)r;
    int N=h[0], nnz=h[1], nstep=h[2];
    int *ptr=malloc((N+1)*sizeof(int)), *col=malloc(nnz*sizeof(int));
    r=fread(ptr,sizeof(int),N+1,f); r=fread(col,sizeof(int),nnz,f);
    double **val=malloc(nstep*sizeof(double*));
    for(int s=0;s<nstep;s++){ val[s]=malloc(nnz*sizeof(double));
                              r=fread(val[s],sizeof(double),nnz,f); }
    double *rhs=malloc(N*sizeof(double)); r=fread(rhs,sizeof(double),N,f);
    double *x=malloc(N*sizeof(double));
    fclose(f);
    printf("  N=%d nnz=%d steps=%d\n\n", N, nnz, nstep);

    AMGX_initialize(); AMGX_register_print_callback(&cb);
    AMGX_config_handle cfg; AMGX_config_create_from_file(&cfg, argv[2]);
    AMGX_resources_handle rs; AMGX_resources_create_simple(&rs,cfg);
    AMGX_matrix_handle A; AMGX_vector_handle B,X; AMGX_solver_handle sv;
    AMGX_matrix_create(&A,rs,AMGX_mode_dDDI);
    AMGX_vector_create(&B,rs,AMGX_mode_dDDI);
    AMGX_vector_create(&X,rs,AMGX_mode_dDDI);
    AMGX_solver_create(&sv,rs,AMGX_mode_dDDI,cfg);
    AMGX_matrix_upload_all(A,N,nnz,1,1,ptr,col,val[0],NULL);
    AMGX_vector_upload(B,N,1,rhs);

    const char *nm[3]={"A full setup each step","B replace + resetup","C replace, reuse hierarchy"};
    for(int mode=0; mode<3; mode++){
        double t_set=0, t_sol=0; int it_tot=0;
        if(mode!=0){   /* modes B and C build ONE hierarchy up front */
            AMGX_matrix_replace_coefficients(A,N,nnz,val[0],NULL);
            AMGX_solver_setup(sv,A);
        }
        for(int s=0;s<nstep;s++){
            double t0=now();
            if(mode==0){
                /* an honest full-setup baseline needs a FRESH solver each step: calling
                   AMGX_solver_setup again on a live handle does not rebuild the hierarchy,
                   which is what made this row report 5,148 it/step in the first attempt. */
                AMGX_solver_destroy(sv);
                AMGX_solver_create(&sv,rs,AMGX_mode_dDDI,cfg);
                AMGX_matrix_replace_coefficients(A,N,nnz,val[s],NULL);
                AMGX_solver_setup(sv,A);
            }
            else if(mode==1){ AMGX_matrix_replace_coefficients(A,N,nnz,val[s],NULL);
                              AMGX_solver_resetup(sv,A); }
            else { AMGX_matrix_replace_coefficients(A,N,nnz,val[s],NULL); }
            t_set += now()-t0;
            for(int i=0;i<N;i++) x[i]=0.0;
            AMGX_vector_upload(X,N,1,x);
            t0=now(); AMGX_solver_solve(sv,B,X); t_sol += now()-t0;
            int it=0; AMGX_solver_get_iterations_number(sv,&it); it_tot+=it;
        }
        printf("  %-28s setup %7.4f s   solve %7.4f s   total %7.4f s   %4d it  (%5.1f/step)\n",
               nm[mode], t_set, t_sol, t_set+t_sol, it_tot, (double)it_tot/nstep);
    }
    AMGX_solver_destroy(sv); AMGX_vector_destroy(X); AMGX_vector_destroy(B);
    AMGX_matrix_destroy(A); AMGX_resources_destroy(rs); AMGX_config_destroy(cfg);
    AMGX_finalize(); return 0;
}
