/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]
#include <stdio.h>

#define mc 256
#define kc 128
#define nb 1000
#define min( i, j ) ( (i)<(j) ? (i): (j) )

/* Routine for computing C = A * B + C */
void AddDot4x4( int k, double* a, int lda, double *b, int ldb, double *c, int ldc);
void AddDot4x4_withoutB( int k, double* a, int lda, double *b, int ldb, double *c, int ldc);
void PackB(int k, double *b, int ldb, double* target);
void InnerKernel( int m, int n, int k, double *a, int lda, 
                                       double *b, int ldb,
                                       double *c, int ldc, int first );
void PackA(int k, double * a, int lda, double * target);

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, p, pb, ib;


  for(p=0; p<k; p+=kc) {
	pb = min(kc, k-p);
    for(i=0; i<m; i+=mc) {
		ib = min(mc, m-i);
    
		InnerKernel(ib, n, pb, &A( i,p ), lda, &B(p, 0 ), ldb, &C( i,0 ), ldc, i==0);
    }
  }
}
void InnerKernel( int m, int n, int k, double *a, int lda, 
                                       double *b, int ldb,
                                       double *c, int ldc,
                                       int first ){
	  int i, j;
    double packedA[m*k];
    static double packedB[kc * nb];
  for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
    //if(first) PackB(k, &B(0, j), ldb, &packedB[j*kc]);
    for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */
      if(j == 0) PackA(k, &A(i, 0), lda, &packedA[i*k]);
      //AddDot4x4( k, &packedA[i*k], 4, &packedB[j*kc], 4, &C( i,j ), ldc );
      AddDot4x4_withoutB( k, &packedA[i*k], 4, &B( 0,j ), ldb, &C( i,j ), ldc );
    }
  }
}

void PackA(int k, double * a, int lda, double * target) {

  int i;
  for(i = 0;i<k;i++) {
    double * pt_a = &A(0, i);
    *target++ = *pt_a;
    *target++ = *(pt_a+1);
    *target++ = *(pt_a+2);
    *target++ = *(pt_a+3);
  }
}
void PackB(int k, double *b, int ldb, double* target) {
  int i;
  double *b00 = &B(0, 0), *b01 = &B(0, 1), *b02 = &B(0,2), *b03 = &B(0, 3);
  for(i = 0;i<k;i++) {
    *target ++ = *b00++;
    *target ++ = *b01++;
    *target ++ = *b02++;
    *target ++ = *b03++;
  }
}
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3
#include <immintrin.h>
typedef union
{
  __m128d v;
  double d[2];
} v2df_t;
void AddDot4x4( int k, double* a, int lda, double *b, int ldb, double *c, int ldc) {
  int p;

  //  AddDot( k, &A( 0, 0 ), lda, &B( 0, 0 ), &C( 0, 0 ) );
  

  v2df_t 
  	c00_10, c01_11, c02_12, c03_13, 

	c20_30, c21_31, c22_32, c23_33,
	a0p_1p, a2p_3p,
	bp0_p0, bp1_p1, bp2_p2, bp3_p3;

    c00_10.v = _mm_setzero_pd(); 
	c01_11.v = _mm_setzero_pd(); 
	c02_12.v = _mm_setzero_pd(); 
	c03_13.v = _mm_setzero_pd(); 
	c20_30.v = _mm_setzero_pd(); 
	c21_31.v = _mm_setzero_pd(); 
	c22_32.v = _mm_setzero_pd(); 
	c23_33.v = _mm_setzero_pd(); 

  for ( p=0; p<k; p++ ){
	
	a0p_1p.v = _mm_load_pd((double *)a);
  a2p_3p.v = _mm_load_pd((double *)(a+2));
  a+=4;

	bp0_p0.v = _mm_loaddup_pd(b);
	bp1_p1.v = _mm_loaddup_pd(b+1);
	bp2_p2.v = _mm_loaddup_pd(b+2);
	bp3_p3.v = _mm_loaddup_pd(b+3);

  b+=4;
/*
	c00_10 += a0p_1p * bp0_p0;
	c01_11 += a0p_1p * bp1_p1;
	c02_12 += a0p_1p * bp2_p2;
	c03_13 += a0p_1p * bp3_p3;
	c20_30 += a2p_3p * bp0_p0;
	c21_31 += a2p_3p * bp1_p1;
	c22_32 += a2p_3p * bp2_p2;
	c23_33 += a2p_3p * bp3_p3;
	*/

	c00_10.v = _mm_fmadd_pd(a0p_1p.v, bp0_p0.v, c00_10.v);
	c01_11.v = _mm_fmadd_pd(a0p_1p.v, bp1_p1.v, c01_11.v);
	c02_12.v = _mm_fmadd_pd(a0p_1p.v, bp2_p2.v, c02_12.v);
	c03_13.v = _mm_fmadd_pd(a0p_1p.v, bp3_p3.v, c03_13.v);

	c20_30.v = _mm_fmadd_pd(a2p_3p.v, bp0_p0.v, c20_30.v);
	c21_31.v = _mm_fmadd_pd(a2p_3p.v, bp1_p1.v, c21_31.v);
	c22_32.v = _mm_fmadd_pd(a2p_3p.v, bp2_p2.v, c22_32.v);
	c23_33.v = _mm_fmadd_pd(a2p_3p.v, bp3_p3.v, c23_33.v);
	
  }
  /*
  _mm_store_pd(&C(0, 0), c00_10.v);
  _mm_store_pd(&C(0, 1), c01_11.v);
  _mm_store_pd(&C(0, 2), c02_12.v);
  _mm_store_pd(&C(0, 3), c03_13.v);
  _mm_store_pd(&C(2, 0), c20_30.v);
  _mm_store_pd(&C(2, 1), c21_31.v);
  _mm_store_pd(&C(2, 2), c22_32.v);
  _mm_store_pd(&C(2, 3), c23_33.v);
  
 */
  C( 0, 0 ) += c00_10.d[0];  C( 0, 1 ) += c01_11.d[0];  
  C( 0, 2 ) += c02_12.d[0];  C( 0, 3 ) += c03_13.d[0]; 

  C( 1, 0 ) += c00_10.d[1];  C( 1, 1 ) += c01_11.d[1];  
  C( 1, 2 ) += c02_12.d[1];  C( 1, 3 ) += c03_13.d[1]; 

  C( 2, 0 ) += c20_30.d[0];  C( 2, 1 ) += c21_31.d[0];  
  C( 2, 2 ) += c22_32.d[0];  C( 2, 3 ) += c23_33.d[0]; 

  C( 3, 0 ) += c20_30.d[1];  C( 3, 1 ) += c21_31.d[1];  
  C( 3, 2 ) += c22_32.d[1];  C( 3, 3 ) += c23_33.d[1]; 

}

void AddDot4x4_withoutB( int k, double* a, int lda, double *b, int ldb, double *c, int ldc) {
  int p;

  //  AddDot( k, &A( 0, 0 ), lda, &B( 0, 0 ), &C( 0, 0 ) );
  
  double *b00, *b01, *b02, *b03;
  b00 = &B(0, 0);
  b01 = &B(0, 1);
  b02 = &B(0, 2);
  b03 = &B(0, 3);
  v2df_t 
  	c00_10, c01_11, c02_12, c03_13, 

	c20_30, c21_31, c22_32, c23_33,
	a0p_1p, a2p_3p,
	bp0_p0, bp1_p1, bp2_p2, bp3_p3;

    c00_10.v = _mm_setzero_pd(); 
	c01_11.v = _mm_setzero_pd(); 
	c02_12.v = _mm_setzero_pd(); 
	c03_13.v = _mm_setzero_pd(); 
	c20_30.v = _mm_setzero_pd(); 
	c21_31.v = _mm_setzero_pd(); 
	c22_32.v = _mm_setzero_pd(); 
	c23_33.v = _mm_setzero_pd(); 

  for ( p=0; p<k; p++ ){
	
	a0p_1p.v = _mm_load_pd((double *)a);
  a2p_3p.v = _mm_load_pd((double *)(a+2));
  a+=4;

	bp0_p0.v = _mm_loaddup_pd(b00++);
	bp1_p1.v = _mm_loaddup_pd(b01++);
	bp2_p2.v = _mm_loaddup_pd(b02++);
	bp3_p3.v = _mm_loaddup_pd(b03++);

/*
	c00_10 += a0p_1p * bp0_p0;
	c01_11 += a0p_1p * bp1_p1;
	c02_12 += a0p_1p * bp2_p2;
	c03_13 += a0p_1p * bp3_p3;
	c20_30 += a2p_3p * bp0_p0;
	c21_31 += a2p_3p * bp1_p1;
	c22_32 += a2p_3p * bp2_p2;
	c23_33 += a2p_3p * bp3_p3;
	*/

	c00_10.v = _mm_fmadd_pd(a0p_1p.v, bp0_p0.v, c00_10.v);
	c01_11.v = _mm_fmadd_pd(a0p_1p.v, bp1_p1.v, c01_11.v);
	c02_12.v = _mm_fmadd_pd(a0p_1p.v, bp2_p2.v, c02_12.v);
	c03_13.v = _mm_fmadd_pd(a0p_1p.v, bp3_p3.v, c03_13.v);

	c20_30.v = _mm_fmadd_pd(a2p_3p.v, bp0_p0.v, c20_30.v);
	c21_31.v = _mm_fmadd_pd(a2p_3p.v, bp1_p1.v, c21_31.v);
	c22_32.v = _mm_fmadd_pd(a2p_3p.v, bp2_p2.v, c22_32.v);
	c23_33.v = _mm_fmadd_pd(a2p_3p.v, bp3_p3.v, c23_33.v);
	
  }
  /*
  _mm_store_pd(&C(0, 0), c00_10.v);
  _mm_store_pd(&C(0, 1), c01_11.v);
  _mm_store_pd(&C(0, 2), c02_12.v);
  _mm_store_pd(&C(0, 3), c03_13.v);
  _mm_store_pd(&C(2, 0), c20_30.v);
  _mm_store_pd(&C(2, 1), c21_31.v);
  _mm_store_pd(&C(2, 2), c22_32.v);
  _mm_store_pd(&C(2, 3), c23_33.v);
  
 */
  C( 0, 0 ) += c00_10.d[0];  C( 0, 1 ) += c01_11.d[0];  
  C( 0, 2 ) += c02_12.d[0];  C( 0, 3 ) += c03_13.d[0]; 

  C( 1, 0 ) += c00_10.d[1];  C( 1, 1 ) += c01_11.d[1];  
  C( 1, 2 ) += c02_12.d[1];  C( 1, 3 ) += c03_13.d[1]; 

  C( 2, 0 ) += c20_30.d[0];  C( 2, 1 ) += c21_31.d[0];  
  C( 2, 2 ) += c22_32.d[0];  C( 2, 3 ) += c23_33.d[0]; 

  C( 3, 0 ) += c20_30.d[1];  C( 3, 1 ) += c21_31.d[1];  
  C( 3, 2 ) += c22_32.d[1];  C( 3, 3 ) += c23_33.d[1]; 

}




  
