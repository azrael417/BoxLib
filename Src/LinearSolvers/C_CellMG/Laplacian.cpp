
#include <winstd.H>
#include <Laplacian.H>
#include <LP_F.H>

Laplacian::Laplacian (const BndryData& bd,
                      Real             _h)
    :
    LinOp(bd,_h) {}

Laplacian::~Laplacian() {}

Real
Laplacian::norm (int nm, int level, const bool local)
{
  switch ( nm )
    {
    case 0:
      return 8.0/(h[level][0]*h[level][0]);
    }
  BoxLib::Error("Bad Laplacian::norm");
  return -1.0;
}

void
Laplacian::compFlux (D_DECL(MultiFab &xflux, MultiFab &yflux, MultiFab &zflux),
		     MultiFab& in, const BC_Mode& bc_mode,
		     int src_comp, int dst_comp, int num_comp, int bnd_comp)
{
    BL_PROFILE("Laplacian::compFlux()");

    const int level    = 0;
    applyBC(in,src_comp,num_comp,level,bc_mode,bnd_comp);

    const bool tiling = true;
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter inmfi(in,tiling); inmfi.isValid(); ++inmfi)
    {
        D_TERM(const Box& xbx   = inmfi.nodaltilebox(0);,
	       const Box& ybx   = inmfi.nodaltilebox(1);,
	       const Box& zbx   = inmfi.nodaltilebox(2););

        FArrayBox& infab = in[inmfi];

        D_TERM(FArrayBox& xfab  = xflux[inmfi];,
               FArrayBox& yfab  = yflux[inmfi];,
               FArrayBox& zfab  = zflux[inmfi];);

        FORT_FLUX(infab.dataPtr(src_comp),
		  ARLIM(infab.loVect()), ARLIM(infab.hiVect()),
		  xbx.loVect(), xbx.hiVect(), 
#if (BL_SPACEDIM >= 2)
		  ybx.loVect(), ybx.hiVect(), 
#if (BL_SPACEDIM == 3)
		  zbx.loVect(), zbx.hiVect(), 
#endif
#endif
	          &num_comp,
		  h[level],
		  xfab.dataPtr(dst_comp),
		  ARLIM(xfab.loVect()), ARLIM(xfab.hiVect())
#if (BL_SPACEDIM >= 2)
		  ,yfab.dataPtr(dst_comp),
		  ARLIM(yfab.loVect()), ARLIM(yfab.hiVect())
#endif
#if (BL_SPACEDIM == 3)
		  ,zfab.dataPtr(dst_comp),
		  ARLIM(zfab.loVect()), ARLIM(zfab.hiVect())
#endif
		  );
    }
}

void
Laplacian::Fsmooth (MultiFab&       solnL,
                    const MultiFab& rhsL,
                    int             level,
                    int             redBlackFlag)
{
    BL_PROFILE("Laplacian::Fsmooth()");

    OrientationIter oitr;

    const FabSet& f0 = (*undrrelxr[level])[oitr()]; oitr++;
    const FabSet& f1 = (*undrrelxr[level])[oitr()]; oitr++;
    const FabSet& f2 = (*undrrelxr[level])[oitr()]; oitr++;
    const FabSet& f3 = (*undrrelxr[level])[oitr()]; oitr++;
#if (BL_SPACEDIM > 2)
    const FabSet& f4 = (*undrrelxr[level])[oitr()]; oitr++;
    const FabSet& f5 = (*undrrelxr[level])[oitr()]; oitr++;
#endif

    oitr.rewind();
    const MultiMask& mm0 = maskvals[level][oitr()]; oitr++;
    const MultiMask& mm1 = maskvals[level][oitr()]; oitr++;
    const MultiMask& mm2 = maskvals[level][oitr()]; oitr++;
    const MultiMask& mm3 = maskvals[level][oitr()]; oitr++;
#if (BL_SPACEDIM > 2)
    const MultiMask& mm4 = maskvals[level][oitr()]; oitr++;
    const MultiMask& mm5 = maskvals[level][oitr()]; oitr++;
#endif

    const int nc = rhsL.nComp();

    const bool tiling = true;

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter solnLmfi(solnL,tiling); solnLmfi.isValid(); ++solnLmfi)
    {
	const Mask& m0 = mm0[solnLmfi];
        const Mask& m1 = mm1[solnLmfi];
        const Mask& m2 = mm2[solnLmfi];
        const Mask& m3 = mm3[solnLmfi];
#if (BL_SPACEDIM > 2)
        const Mask& m4 = mm4[solnLmfi];
        const Mask& m5 = mm5[solnLmfi];
#endif

		const Box&       tbx     = solnLmfi.tilebox();
        const Box&       vbx     = solnLmfi.validbox();
		const Box&       fbx     = solnLmfi.fabbox();
        FArrayBox&       solnfab = solnL[solnLmfi];
        const FArrayBox& rhsfab  = rhsL[solnLmfi];
        const FArrayBox& f0fab   = f0[solnLmfi];
        const FArrayBox& f1fab   = f1[solnLmfi];
        const FArrayBox& f2fab   = f2[solnLmfi];
        const FArrayBox& f3fab   = f3[solnLmfi];
#if (BL_SPACEDIM == 3)
        const FArrayBox& f4fab   = f4[solnLmfi];
        const FArrayBox& f5fab   = f5[solnLmfi];
#endif

#if (BL_SPACEDIM == 2)
        FORT_GSRB(
            solnfab.dataPtr(), 
            ARLIM(solnfab.loVect()),ARLIM(solnfab.hiVect()),
            rhsfab.dataPtr(), 
            ARLIM(rhsfab.loVect()), ARLIM(rhsfab.hiVect()),
            f0fab.dataPtr(), 
            ARLIM(f0fab.loVect()), ARLIM(f0fab.hiVect()),
            m0.dataPtr(), 
            ARLIM(m0.loVect()), ARLIM(m0.hiVect()),
            f1fab.dataPtr(), 
            ARLIM(f1fab.loVect()), ARLIM(f1fab.hiVect()),
            m1.dataPtr(), 
            ARLIM(m1.loVect()), ARLIM(m1.hiVect()),
            f2fab.dataPtr(), 
            ARLIM(f2fab.loVect()), ARLIM(f2fab.hiVect()),
            m2.dataPtr(), 
            ARLIM(m2.loVect()), ARLIM(m2.hiVect()),
            f3fab.dataPtr(), 
            ARLIM(f3fab.loVect()), ARLIM(f3fab.hiVect()),
            m3.dataPtr(), 
            ARLIM(m3.loVect()), ARLIM(m3.hiVect()),
	    tbx.loVect(), tbx.hiVect(), vbx.loVect(), vbx.hiVect(),
            &nc, h[level], &redBlackFlag);
#endif

#if (BL_SPACEDIM == 3)
//#ifdef USE_CPP_KERNELS
//	
//	//static inline int index3(const Box& field, const int& i, const int& j, const int& k, const int& BL_ghosts){
//	//	const int BL_jStride = field.tilebox.length(0);
//	//	const int BL_kStride = field.tilebox.length(0) * field.tilebox.length(1);
//	//	
//	//	return (i+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
//	//}
//		
//	Real gamma=6.;
//	Real hl=h[level][0];
//	
//	const int* lo = tbx.loVect();
//	const int* hi = tbx.hiVect();
//	const int* blo = vbx.loVect();
//	const int* bhi = vbx.hiVect();
//
//	const int BL_jStride = tbx.length(0);
//	const int BL_kStride = tbx.length(0) * tbx.length(1);
//	const int BL_ghosts = solnL.nGrow();
//	const int offset = fbx.length(0)*fbx.length(1)*fbx.length(2);
//	
//	//pointers
//	Real* phip = solnfab.dataPtr();
//	const Real* rhsp = rhsfab.dataPtr();
//	const int* m0p = m0.dataPtr();
//	const int* m1p = m1.dataPtr();
//	const int* m2p = m2.dataPtr();
//	const int* m3p = m3.dataPtr();
//	const int* m4p = m4.dataPtr();
//	const int* m5p = m5.dataPtr();
//	const Real* f0p = f0fab.dataPtr();
//	const Real* f1p = f1fab.dataPtr();
//	const Real* f2p = f2fab.dataPtr();
//	const Real* f3p = f3fab.dataPtr();
//	const Real* f4p = f4fab.dataPtr();
//	const Real* f5p = f5fab.dataPtr();
//	
//	std::cout << "Using C++ GSRB" << std::endl;
//	
//	for(unsigned int n=0; n<nc; n++){
//	// do n = 1, nc
//		for(unsigned int k=lo[2]; k<=hi[2]; k++){
//		//do k = lo(3), hi(3)
//			for(unsigned int j=lo[1]; j<=hi[1]; j++){
//			//do j = lo(2), hi(2)
//				unsigned int ioff = (lo[0] + j + k + redBlackFlag)%2;
//				//ioff = MOD(lo(1) + j + k + redblack,2)
//				for(unsigned int i=ioff+lo[0]; i<=hi[0]; i+=2){
//				//do i = lo(1) + ioff,hi(1),2
//						
//					//coordinates
//					int   i_j_k = (i+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
//					int im1_j_k = (i-1+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
//					int ip1_j_k = (i+1+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
//					int i_jm1_k = (i+BL_ghosts) + (j-1+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
//					int i_jp1_k = (i+BL_ghosts) + (j+1+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
//					int i_j_km1 = (i+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (k-1+BL_ghosts)*BL_kStride;
//					int i_j_kp1 = (i+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (k+1+BL_ghosts)*BL_kStride;
//					
//					//boundary
//					int blo_j_k = (blo[0]+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
//					int blom1_j_k = (blo[0]-1+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
//					double cf0 = ( (i==blo[0] && m0p[blom1_j_k] > 0) ? f0p[blo_j_k] : 0.);
//					//merge(f0(blo(1),j,k), 0.0D0, (i .eq. blo(1)) .and. (m0(blo(1)-1,j,k).gt.0));
//					int i_blo_k = (i+BL_ghosts) + (blo[1]+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
//					int i_blom1_k = (i+BL_ghosts) + (blo[1]-1+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
//					double cf1 = ( (j==blo[1] && m1p[i_blom1_k] > 0) ? f1p[i_blo_k] : 0.);
//					//merge(f1(i,blo(2),k), 0.0D0, (j .eq. blo(2)) .and. (m1(i,blo(2)-1,k).gt.0));
//					int i_j_blo = (i+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (blo[2]+BL_ghosts)*BL_kStride;
//					int i_j_blom1 = (i+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (blo[2]-1+BL_ghosts)*BL_kStride;
//					double cf2 =  ( (k==blo[2] && m2p[i_j_blom1] > 0) ? f2p[i_j_blo] : 0.);
//					//merge(f2(i,j,blo(3)), 0.0D0, (k .eq. blo(3)) .and. (m2(i,j,blo(3)-1).gt.0));
//					int bhi_j_k = (bhi[0]+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
//					int bhip1_j_k = (bhi[0]+1+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
//					double cf3 = ( (i==bhi[0] && m3p[bhip1_j_k] > 0) ? f3p[bhi_j_k] : 0.);
//					//merge(f3(bhi(1),j,k), 0.0D0, (i .eq. bhi(1)) .and. (m3(bhi(1)+1,j,k).gt.0));
//					int i_bhi_k = (i+BL_ghosts) + (bhi[1]+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
//					int i_bhip1_k = (i+BL_ghosts) + (bhi[1]+1+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
//					double cf4 = ( (j==bhi[1] && m4p[i_bhip1_k] > 0) ? f4p[i_bhi_k] : 0.);
//					//merge(f4(i,bhi(2),k), 0.0D0, (j .eq. bhi(2)) .and. (m4(i,bhi(2)+1,k).gt.0));
//					int i_j_bhi = (i+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (bhi[2]+BL_ghosts)*BL_kStride;
//					int i_j_bhip1 = (i+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (bhi[2]+1+BL_ghosts)*BL_kStride;
//					double cf5 = ( (k==bhi[2] && m5p[i_j_bhip1] > 0) ? f5p[i_j_bhi] : 0.);
//					//merge(f5(i,j,bhi(3)), 0.0D0, (k .eq. bhi(3)) .and. (m5(i,j,bhi(3)+1).gt.0));
//					
//					//combine boundary terms
//					double delta = cf0 + cf1 + cf2 + cf3 + cf4 + cf5;
//					//compute rho:
//					double rho =  phip[im1_j_k+offset*n] + phip[ip1_j_k+offset*n] + phip[i_jm1_k+offset*n] + phip[i_jp1_k+offset*n] + phip[i_j_km1+offset*n] + phip[i_j_kp1 + offset*n];
//					//phi(i-1,j,k,n) + phi(i+1,j,k,n) + phi(i,j-1,k,n) + phi(i,j+1,k,n) + phi(i,j,k-1,n) + phi(i,j,k+1,n);
//					phip[i_j_k+offset*n] = (rhsp[i_j_k+offset*n]*hl*hl - rho + phip[i_j_k+offset*n]*delta)/(delta - gamma);
//					//phi(i,j,k,n) = (rhs(i,j,k,n)*h*h - rho + phi(i,j,k,n)*delta)/(delta - gamma);
//				}
//			}
//		}
//	}
//#else
        FORT_GSRB(
            solnfab.dataPtr(), 
            ARLIM(solnfab.loVect()),ARLIM(solnfab.hiVect()),
            rhsfab.dataPtr(), 
            ARLIM(rhsfab.loVect()), ARLIM(rhsfab.hiVect()),
            f0fab.dataPtr(), 
            ARLIM(f0fab.loVect()), ARLIM(f0fab.hiVect()),
            m0.dataPtr(), 
            ARLIM(m0.loVect()), ARLIM(m0.hiVect()),
            f1fab.dataPtr(), 
            ARLIM(f1fab.loVect()), ARLIM(f1fab.hiVect()),
            m1.dataPtr(), 
            ARLIM(m1.loVect()), ARLIM(m1.hiVect()),
            f2fab.dataPtr(), 
            ARLIM(f2fab.loVect()), ARLIM(f2fab.hiVect()),
            m2.dataPtr(), 
            ARLIM(m2.loVect()), ARLIM(m2.hiVect()),
            f3fab.dataPtr(), 
            ARLIM(f3fab.loVect()), ARLIM(f3fab.hiVect()),
            m3.dataPtr(), 
            ARLIM(m3.loVect()), ARLIM(m3.hiVect()),
            f4fab.dataPtr(), 
            ARLIM(f4fab.loVect()), ARLIM(f4fab.hiVect()),
            m4.dataPtr(), 
            ARLIM(m4.loVect()), ARLIM(m4.hiVect()),
            f5fab.dataPtr(), 
            ARLIM(f5fab.loVect()), ARLIM(f5fab.hiVect()),
            m5.dataPtr(), 
            ARLIM(m5.loVect()), ARLIM(m5.hiVect()),
	    tbx.loVect(), tbx.hiVect(), vbx.loVect(), vbx.hiVect(),
	    &nc, h[level], &redBlackFlag);
		//#endif
#endif
    }
}

void
Laplacian::Fsmooth_jacobi (MultiFab&       solnL,
                           const MultiFab& rhsL,
                           int            level)
{
}

void
Laplacian::Fapply (MultiFab&       y,
                   const MultiFab& x,
                   int             level)
{
  int src_comp = 0;
  int dst_comp = 0;
  int num_comp = 1;
  Fapply(y,dst_comp,x,src_comp,num_comp,level);
}

void
Laplacian::Fapply (MultiFab&       y,
		   int             dst_comp,
                   const MultiFab& x,
		   int             src_comp,
		   int             num_comp,
                   int             level)
{
    BL_PROFILE("Laplacian::Fapply()");

    const bool tiling = true;
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter ymfi(y,tiling); ymfi.isValid(); ++ymfi)
    {
        const Box&       tbx  = ymfi.tilebox();
        FArrayBox&       yfab = y[ymfi];
        const FArrayBox& xfab = x[ymfi];

        FORT_ADOTX(yfab.dataPtr(dst_comp), 
                   ARLIM(yfab.loVect()), ARLIM(yfab.hiVect()),
                   xfab.dataPtr(src_comp), 
                   ARLIM(xfab.loVect()), ARLIM(xfab.hiVect()),
                   tbx.loVect(), tbx.hiVect(), &num_comp,
                   h[level]);
    }
}
