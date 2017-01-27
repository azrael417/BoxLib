#include <winstd.H>
#include <algorithm>
#include <ABecLaplacian.H>
#include <ABec_F.H>
#include <ParallelDescriptor.H>

Real ABecLaplacian::a_def     = 0.0;
Real ABecLaplacian::b_def     = 1.0;
Real ABecLaplacian::alpha_def = 1.0;
Real ABecLaplacian::beta_def  = 1.0;

ABecLaplacian::ABecLaplacian (const BndryData& _bd,
                              Real             _h)
    :
    LinOp(_bd,_h),
    alpha(alpha_def),
    beta(beta_def)
{
    initCoefficients(_bd.boxes());
}

ABecLaplacian::ABecLaplacian (const BndryData& _bd,
                              const Real*      _h)
    :
    LinOp(_bd,_h),
    alpha(alpha_def),
    beta(beta_def)
{
    initCoefficients(_bd.boxes());
}

ABecLaplacian::ABecLaplacian (BndryData*  _bd,
                              const Real* _h)
    :
    LinOp(_bd,_h),
    alpha(alpha_def),
    beta(beta_def)
{
    initCoefficients(_bd->boxes());
}

ABecLaplacian::~ABecLaplacian ()
{
    clearToLevel(-1);
}

Real
ABecLaplacian::norm (int nm, int level, const bool local)
{
    BL_PROFILE("ABecLaplacian::norm()");

    BL_ASSERT(nm == 0);
    const MultiFab& a   = aCoefficients(level);

    D_TERM(const MultiFab& bX  = bCoefficients(0,level);,
           const MultiFab& bY  = bCoefficients(1,level);,
           const MultiFab& bZ  = bCoefficients(2,level););

    //const int nc = a.nComp(); // FIXME: This LinOp only really support single-component
    const int nc = 1;
    Real res = 0.0;

    const bool tiling = true;

#ifdef _OPENMP
#pragma omp parallel reduction(max:res)
#endif
    {
	for (MFIter amfi(a,tiling); amfi.isValid(); ++amfi)
	{
	    Real tres;
	    
	    const Box&       tbx  = amfi.tilebox();
	    const FArrayBox& afab = a[amfi];
	    
	    D_TERM(const FArrayBox& bxfab = bX[amfi];,
		   const FArrayBox& byfab = bY[amfi];,
		   const FArrayBox& bzfab = bZ[amfi];);
	    
#if (BL_SPACEDIM==2)
	    FORT_NORMA(&tres,
		       &alpha, &beta,
		       afab.dataPtr(),  ARLIM(afab.loVect()), ARLIM(afab.hiVect()),
		       bxfab.dataPtr(), ARLIM(bxfab.loVect()), ARLIM(bxfab.hiVect()),
		       byfab.dataPtr(), ARLIM(byfab.loVect()), ARLIM(byfab.hiVect()),
		       tbx.loVect(), tbx.hiVect(), &nc,
		       h[level]);
#elif (BL_SPACEDIM==3)
	    
	    FORT_NORMA(&tres,
		       &alpha, &beta,
		       afab.dataPtr(),  ARLIM(afab.loVect()), ARLIM(afab.hiVect()),
		       bxfab.dataPtr(), ARLIM(bxfab.loVect()), ARLIM(bxfab.hiVect()),
		       byfab.dataPtr(), ARLIM(byfab.loVect()), ARLIM(byfab.hiVect()),
		       bzfab.dataPtr(), ARLIM(bzfab.loVect()), ARLIM(bzfab.hiVect()),
		       tbx.loVect(), tbx.hiVect(), &nc,
		       h[level]);
#endif

	    res = std::max(res, tres);
	}
    }

    if (!local)
        ParallelDescriptor::ReduceRealMax(res,color());
    return res;
}

void
ABecLaplacian::clearToLevel (int level)
{
  BL_ASSERT(level >= -1);

  for (int i = level+1; i < numLevels(); ++i)
  {
    if (acoefs[i] != 0) {
      delete acoefs[i];
      acoefs[i] = 0;
    }
    a_valid[i] = false;

    for (int j = 0; j < BL_SPACEDIM; ++j)
    {
      if (bcoefs[i][j] != 0) {
        delete bcoefs[i][j];
        bcoefs[i][j] = 0;
      }
    }
    b_valid[i] = false;
  }
}

void
ABecLaplacian::prepareForLevel (int level)
{
    LinOp::prepareForLevel(level);

    if (level == 0 )
        return;

    prepareForLevel(level-1);
    //
    // If coefficients were marked invalid, or if not yet made, make new ones
    // (Note: makeCoefficients is a LinOp routine, and it allocates AND
    // fills coefficients.  A more efficient implementation would allocate
    // and fill in separate steps--we could then use the a_valid bool
    // along with the length of a_valid to separately determine whether to
    // fill or allocate the coefficient MultiFabs.
    //
    if (level >= a_valid.size() || a_valid[level] == false)
    {
        if (acoefs.size() < level+1)
        {
            acoefs.resize(level+1);
            acoefs[level] = new MultiFab;
        }
        else
        {
            delete acoefs[level];
            acoefs[level] = new MultiFab;
        }
        makeCoefficients(*acoefs[level], *acoefs[level-1], level);
        a_valid.resize(level+1);
        a_valid[level] = true;
    }
    
    if (level >= b_valid.size() || b_valid[level] == false)
    {
        if (bcoefs.size() < level+1)
        {
            bcoefs.resize(level+1);
            for(int i = 0; i < BL_SPACEDIM; ++i)
                bcoefs[level][i] = new MultiFab;
        }
        else
        {
            for(int i = 0; i < BL_SPACEDIM; ++i)
            {
                delete bcoefs[level][i];
                bcoefs[level][i] = new MultiFab;
            }
        }
        for (int i = 0; i < BL_SPACEDIM; ++i)
        {
            makeCoefficients(*bcoefs[level][i], *bcoefs[level-1][i], level);
        }
        b_valid.resize(level+1);
        b_valid[level] = true;
    }
}

void
ABecLaplacian::initCoefficients (const BoxArray& _ba)
{
    const int nComp=1;
    const int nGrow=0;
    ParallelDescriptor::Color clr = color();
    acoefs.resize(1);
    bcoefs.resize(1);
    acoefs[0] = new MultiFab(_ba, nComp, nGrow, clr);
    acoefs[0]->setVal(a_def);
    a_valid.resize(1);
    a_valid[0] = true;

    for (int i = 0; i < BL_SPACEDIM; ++i)
    {
        BoxArray edge_boxes(_ba);
        edge_boxes.surroundingNodes(i);
        bcoefs[0][i] = new MultiFab(edge_boxes, nComp, nGrow, clr);
        bcoefs[0][i]->setVal(b_def);
    }
    b_valid.resize(1);
    b_valid[0] = true;
}

void
ABecLaplacian::aCoefficients (const MultiFab& _a)
{
    BL_ASSERT(_a.ok());
    BL_ASSERT(_a.boxArray() == (acoefs[0])->boxArray());
    invalidate_a_to_level(0);
    (*acoefs[0]).copy(_a,0,0,1);
}

void
ABecLaplacian::ZeroACoefficients ()
{
    invalidate_a_to_level(0);
    (*acoefs[0]).setVal(0,0,acoefs[0]->nComp(),acoefs[0]->nGrow());
}

void
ABecLaplacian::bCoefficients (const MultiFab& _b,
                              int             dir)
{
    BL_ASSERT(_b.ok());
    BL_ASSERT(_b.boxArray() == (bcoefs[0][dir])->boxArray());
    invalidate_b_to_level(0);
    (*bcoefs[0][dir]).copy(_b,0,0,1);
}

void
ABecLaplacian::bCoefficients (const FArrayBox& _b,
                              int              dir,
                              int              gridno)
{
    BL_ASSERT(_b.box().contains((bcoefs[0][dir])->boxArray()[gridno]));
    invalidate_b_to_level(0);
    (*bcoefs[0][dir])[gridno].copy(_b,0,0,1);
}

const MultiFab&
ABecLaplacian::aCoefficients (int level)
{
    prepareForLevel(level);
    return *acoefs[level];
}

const MultiFab&
ABecLaplacian::bCoefficients (int dir,int level)
{
    prepareForLevel(level);
    return *bcoefs[level][dir];
}

void
ABecLaplacian::setCoefficients (const MultiFab &_a,
                                const MultiFab &_bX,
                                const MultiFab &_bY)
{
    aCoefficients(_a);
    bCoefficients(_bX, 0);
    bCoefficients(_bY, 1);
}

void
ABecLaplacian::setCoefficients (const MultiFab& _a,
                                const MultiFab* _b)
{
    aCoefficients(_a);
    for (int n = 0; n < BL_SPACEDIM; ++n)
        bCoefficients(_b[n], n);
}

void
ABecLaplacian::setCoefficients (const MultiFab& _a,
                                const PArray<MultiFab>& _b)
{
    aCoefficients(_a);
    for (int n = 0; n < BL_SPACEDIM; ++n)
        bCoefficients(_b[n], n);
}

void
ABecLaplacian::invalidate_a_to_level (int lev)
{
    lev = (lev >= 0 ? lev : 0);
    for (int i = lev; i < numLevels(); i++)
        a_valid[i] = false;
}

void
ABecLaplacian::invalidate_b_to_level (int lev)
{
    lev = (lev >= 0 ? lev : 0);
    for (int i = lev; i < numLevels(); i++)
        b_valid[i] = false;
}

void
ABecLaplacian::compFlux (D_DECL(MultiFab &xflux, MultiFab &yflux, MultiFab &zflux),
			 MultiFab& in, const BC_Mode& bc_mode,
			 int src_comp, int dst_comp, int num_comp, int bnd_comp)
{
  compFlux(D_DECL(xflux, yflux, zflux), in, true, bc_mode, src_comp, dst_comp, num_comp, bnd_comp);
}

void
ABecLaplacian::compFlux (D_DECL(MultiFab &xflux, MultiFab &yflux, MultiFab &zflux),
                         MultiFab& in, bool do_ApplyBC, const BC_Mode& bc_mode,
			 int src_comp, int dst_comp, int num_comp, int bnd_comp)
{
    BL_PROFILE("ABecLaplacian::compFlux()");

    const int level = 0;
    BL_ASSERT(num_comp==1);

    if (do_ApplyBC)
      applyBC(in,src_comp,num_comp,level,bc_mode,bnd_comp);

    const MultiFab& a = aCoefficients(level);

    D_TERM(const MultiFab& bX = bCoefficients(0,level);,
           const MultiFab& bY = bCoefficients(1,level);,
           const MultiFab& bZ = bCoefficients(2,level););

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

        D_TERM(const FArrayBox& bxfab = bX[inmfi];,
               const FArrayBox& byfab = bY[inmfi];,
               const FArrayBox& bzfab = bZ[inmfi];);

        D_TERM(FArrayBox& xfluxfab = xflux[inmfi];,
               FArrayBox& yfluxfab = yflux[inmfi];,
               FArrayBox& zfluxfab = zflux[inmfi];);

        FORT_FLUX(infab.dataPtr(src_comp),
		  ARLIM(infab.loVect()), ARLIM(infab.hiVect()),
		  &alpha, &beta, a[inmfi].dataPtr(), 
		  ARLIM(a[inmfi].loVect()), ARLIM(a[inmfi].hiVect()),
		  bxfab.dataPtr(), 
		  ARLIM(bxfab.loVect()), ARLIM(bxfab.hiVect()),
#if (BL_SPACEDIM >= 2)
		  byfab.dataPtr(), 
		  ARLIM(byfab.loVect()), ARLIM(byfab.hiVect()),
#if (BL_SPACEDIM == 3)
		  bzfab.dataPtr(), 
		  ARLIM(bzfab.loVect()), ARLIM(bzfab.hiVect()),
#endif
#endif
		  xbx.loVect(), xbx.hiVect(), 
#if (BL_SPACEDIM >= 2)
		  ybx.loVect(), ybx.hiVect(), 
#if (BL_SPACEDIM == 3)
		  zbx.loVect(), zbx.hiVect(), 
#endif
#endif
		  &num_comp,
		  h[level],
		  xfluxfab.dataPtr(dst_comp),
		  ARLIM(xfluxfab.loVect()), ARLIM(xfluxfab.hiVect())
#if (BL_SPACEDIM >= 2)
		  ,yfluxfab.dataPtr(dst_comp),
		  ARLIM(yfluxfab.loVect()), ARLIM(yfluxfab.hiVect())
#endif
#if (BL_SPACEDIM == 3)
		  ,zfluxfab.dataPtr(dst_comp),
		  ARLIM(zfluxfab.loVect()), ARLIM(zfluxfab.hiVect())
#endif
		  );
    }
}
        
//
// Must be defined for MultiGrid/CGSolver to work.
//

void
ABecLaplacian::Fsmooth (MultiFab&       solnL,
                        const MultiFab& rhsL,
                        int             level,
                        int             redBlackFlag)
{
    BL_PROFILE("ABecLaplacian::Fsmooth()");

    OrientationIter oitr;

    const FabSet& f0 = (*undrrelxr[level])[oitr()]; oitr++;
    const FabSet& f1 = (*undrrelxr[level])[oitr()]; oitr++;
    const FabSet& f2 = (*undrrelxr[level])[oitr()]; oitr++;
    const FabSet& f3 = (*undrrelxr[level])[oitr()]; oitr++;
#if (BL_SPACEDIM > 2)
    const FabSet& f4 = (*undrrelxr[level])[oitr()]; oitr++;
    const FabSet& f5 = (*undrrelxr[level])[oitr()]; oitr++;
#endif    
    const MultiFab& a = aCoefficients(level);

    D_TERM(const MultiFab& bX = bCoefficients(0,level);,
           const MultiFab& bY = bCoefficients(1,level);,
           const MultiFab& bZ = bCoefficients(2,level););

    oitr.rewind();
    const MultiMask& mm0 = maskvals[level][oitr()]; oitr++;
    const MultiMask& mm1 = maskvals[level][oitr()]; oitr++;
    const MultiMask& mm2 = maskvals[level][oitr()]; oitr++;
    const MultiMask& mm3 = maskvals[level][oitr()]; oitr++;
#if (BL_SPACEDIM > 2)
    const MultiMask& mm4 = maskvals[level][oitr()]; oitr++;
    const MultiMask& mm5 = maskvals[level][oitr()]; oitr++;
#endif

    //const int nc = solnL.nComp(); // FIXME: This LinOp only really supports single-component
    const int nc = 1;

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
        const FArrayBox& afab    = a[solnLmfi];

        D_TERM(const FArrayBox& bxfab = bX[solnLmfi];,
               const FArrayBox& byfab = bY[solnLmfi];,
               const FArrayBox& bzfab = bZ[solnLmfi];);

        const FArrayBox& f0fab = f0[solnLmfi];
        const FArrayBox& f1fab = f1[solnLmfi];
        const FArrayBox& f2fab = f2[solnLmfi];
        const FArrayBox& f3fab = f3[solnLmfi];
#if (BL_SPACEDIM > 2)
        const FArrayBox& f4fab = f4[solnLmfi];
        const FArrayBox& f5fab = f5[solnLmfi];
#endif

#if (BL_SPACEDIM == 2)
        FORT_GSRB(solnfab.dataPtr(), ARLIM(solnfab.loVect()),ARLIM(solnfab.hiVect()),
                  rhsfab.dataPtr(), ARLIM(rhsfab.loVect()), ARLIM(rhsfab.hiVect()),
                  &alpha, &beta,
                  afab.dataPtr(), ARLIM(afab.loVect()),    ARLIM(afab.hiVect()),
                  bxfab.dataPtr(), ARLIM(bxfab.loVect()),   ARLIM(bxfab.hiVect()),
                  byfab.dataPtr(), ARLIM(byfab.loVect()),   ARLIM(byfab.hiVect()),
                  f0fab.dataPtr(), ARLIM(f0fab.loVect()),   ARLIM(f0fab.hiVect()),
                  m0.dataPtr(), ARLIM(m0.loVect()),   ARLIM(m0.hiVect()),
                  f1fab.dataPtr(), ARLIM(f1fab.loVect()),   ARLIM(f1fab.hiVect()),
                  m1.dataPtr(), ARLIM(m1.loVect()),   ARLIM(m1.hiVect()),
                  f2fab.dataPtr(), ARLIM(f2fab.loVect()),   ARLIM(f2fab.hiVect()),
                  m2.dataPtr(), ARLIM(m2.loVect()),   ARLIM(m2.hiVect()),
                  f3fab.dataPtr(), ARLIM(f3fab.loVect()),   ARLIM(f3fab.hiVect()),
                  m3.dataPtr(), ARLIM(m3.loVect()),   ARLIM(m3.hiVect()),
                  tbx.loVect(), tbx.hiVect(), vbx.loVect(), vbx.hiVect(),
                  &nc, h[level], &redBlackFlag);
#endif

#if (BL_SPACEDIM == 3)
#ifdef USE_CPP_KERNELS
	
	//static inline int index3(const Box& field, const int& i, const int& j, const int& k, const int& BL_ghosts){
	//	const int BL_jStride = field.tilebox.length(0);
	//	const int BL_kStride = field.tilebox.length(0) * field.tilebox.length(1);
	//	
	//	return (i+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
	//}
		
	Real omega=1.15;
	Tuple<Real,3> hl=h[level];
	
	const int* lo = tbx.loVect();
	const int* hi = tbx.hiVect();
	const int* blo = vbx.loVect();
	const int* bhi = vbx.hiVect();

	const int BL_jStride = tbx.length(0);
	const int BL_kStride = tbx.length(0) * tbx.length(1);
	const int BL_ghosts = solnL.nGrow();
	const int offset = fbx.length(0)*fbx.length(1)*fbx.length(2);
	
	//pointers
	Real* phip = solnfab.dataPtr();
	const Real* rhsp = rhsfab.dataPtr();
	const Real* ap = afab.dataPtr();
	const Real* bxp = bxfab.dataPtr();
	const Real* byp = byfab.dataPtr();
	const Real* bzp = bzfab.dataPtr();
	const int* m0p = m0.dataPtr();
	const int* m1p = m1.dataPtr();
	const int* m2p = m2.dataPtr();
	const int* m3p = m3.dataPtr();
	const int* m4p = m4.dataPtr();
	const int* m5p = m5.dataPtr();
	const Real* f0p = f0fab.dataPtr();
	const Real* f1p = f1fab.dataPtr();
	const Real* f2p = f2fab.dataPtr();
	const Real* f3p = f3fab.dataPtr();
	const Real* f4p = f4fab.dataPtr();
	const Real* f5p = f5fab.dataPtr();
	
	//constants
    double dhx = beta/(hl[0]*hl[0]);
    double dhy = beta/(hl[1]*hl[1]);
    double dhz = beta/(hl[2]*hl[2]);

	// loop over sources
	for(unsigned int n=0; n<nc; n++){
	// do n = 1, nc
		for(unsigned int k=lo[2]; k<=hi[2]; k++){
		//do k = lo(3), hi(3)
			for(unsigned int j=lo[1]; j<=hi[1]; j++){
			//do j = lo(2), hi(2)
				unsigned int ioff = (lo[0] + j + k + redBlackFlag)%2;
				//ioff = MOD(lo(1) + j + k + redblack,2)
				for(unsigned int i=ioff+lo[0]; i<=hi[0]; i+=2){
				//do i = lo(1) + ioff,hi(1),2
						
					//coordinates
					int   i_j_k = (i+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
					int im1_j_k = (i-1+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
					int ip1_j_k = (i+1+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
					int i_jm1_k = (i+BL_ghosts) + (j-1+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
					int i_jp1_k = (i+BL_ghosts) + (j+1+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
					int i_j_km1 = (i+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (k-1+BL_ghosts)*BL_kStride;
					int i_j_kp1 = (i+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (k+1+BL_ghosts)*BL_kStride;
					
					//boundary
					int blo_j_k = (blo[0]+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
					int blom1_j_k = (blo[0]-1+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
					double cf0 = ( (i==blo[0] && m0p[blom1_j_k] > 0) ? f0p[blo_j_k] : 0.);
					//merge(f0(blo(1),j,k), 0.0D0, (i .eq. blo(1)) .and. (m0(blo(1)-1,j,k).gt.0));
					int i_blo_k = (i+BL_ghosts) + (blo[1]+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
					int i_blom1_k = (i+BL_ghosts) + (blo[1]-1+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
					double cf1 = ( (j==blo[1] && m1p[i_blom1_k] > 0) ? f1p[i_blo_k] : 0.);
					//merge(f1(i,blo(2),k), 0.0D0, (j .eq. blo(2)) .and. (m1(i,blo(2)-1,k).gt.0));
					int i_j_blo = (i+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (blo[2]+BL_ghosts)*BL_kStride;
					int i_j_blom1 = (i+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (blo[2]-1+BL_ghosts)*BL_kStride;
					double cf2 =  ( (k==blo[2] && m2p[i_j_blom1] > 0) ? f2p[i_j_blo] : 0.);
					//merge(f2(i,j,blo(3)), 0.0D0, (k .eq. blo(3)) .and. (m2(i,j,blo(3)-1).gt.0));
					int bhi_j_k = (bhi[0]+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
					int bhip1_j_k = (bhi[0]+1+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
					double cf3 = ( (i==bhi[0] && m3p[bhip1_j_k] > 0) ? f3p[bhi_j_k] : 0.);
					//merge(f3(bhi(1),j,k), 0.0D0, (i .eq. bhi(1)) .and. (m3(bhi(1)+1,j,k).gt.0));
					int i_bhi_k = (i+BL_ghosts) + (bhi[1]+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
					int i_bhip1_k = (i+BL_ghosts) + (bhi[1]+1+BL_ghosts)*BL_jStride + (k+BL_ghosts)*BL_kStride;
					double cf4 = ( (j==bhi[1] && m4p[i_bhip1_k] > 0) ? f4p[i_bhi_k] : 0.);
					//merge(f4(i,bhi(2),k), 0.0D0, (j .eq. bhi(2)) .and. (m4(i,bhi(2)+1,k).gt.0));
					int i_j_bhi = (i+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (bhi[2]+BL_ghosts)*BL_kStride;
					int i_j_bhip1 = (i+BL_ghosts) + (j+BL_ghosts)*BL_jStride + (bhi[2]+1+BL_ghosts)*BL_kStride;
					double cf5 = ( (k==bhi[2] && m5p[i_j_bhip1] > 0) ? f5p[i_j_bhi] : 0.);
					//merge(f5(i,j,bhi(3)), 0.0D0, (k .eq. bhi(3)) .and. (m5(i,j,bhi(3)+1).gt.0));
					
					//compute gamma:
					Real gamma = alpha*ap[i_j_k];
					gamma += dhx*(bxp[i_j_k]+bxp[ip1_j_k]);
					gamma += dhy*(byp[i_j_k]+byp[i_jp1_k]);
					gamma += dhz*(bzp[i_j_k]+bzp[i_j_kp1]);
					
					//compute g_m_d:
					Real g_m_d = gamma;
					g_m_d -= dhx*(bxp[i_j_k]*cf0+bxp[ip1_j_k]*cf3);
					g_m_d -= dhy*(byp[i_j_k]*cf1+byp[i_jp1_k]*cf4);
					g_m_d -= dhz*(bzp[i_j_k]*cf2+bzp[i_j_kp1]*cf5);
					
					//compute rho:
					Real rho;
					rho =  dhx * (bxp[i_j_k] * phip[im1_j_k+offset*n] + bxp[ip1_j_k]*phip[ip1_j_k+offset*n]);
					rho += dhy * (byp[i_j_k] * phip[i_jm1_k+offset*n] + byp[i_jp1_k]*phip[i_jp1_k+offset*n]);
					rho += dhz * (bzp[i_j_k] * phip[i_j_km1+offset*n] + bzp[i_j_kp1]*phip[i_j_kp1+offset*n]);
					
					//compute res:
					Real res = rhsp[i_j_k+offset*n] - (gamma*phip[i_j_k+offset*n] - rho);
					
					//update phi
					phip[i_j_k+offset*n] +=  omega/g_m_d * res;
				}
			}
		}
	}
#else
        FORT_GSRB(solnfab.dataPtr(), ARLIM(solnfab.loVect()),ARLIM(solnfab.hiVect()),
                  rhsfab.dataPtr(), ARLIM(rhsfab.loVect()), ARLIM(rhsfab.hiVect()),
                  &alpha, &beta,
                  afab.dataPtr(), ARLIM(afab.loVect()), ARLIM(afab.hiVect()),
                  bxfab.dataPtr(), ARLIM(bxfab.loVect()), ARLIM(bxfab.hiVect()),
                  byfab.dataPtr(), ARLIM(byfab.loVect()), ARLIM(byfab.hiVect()),
                  bzfab.dataPtr(), ARLIM(bzfab.loVect()), ARLIM(bzfab.hiVect()),
                  f0fab.dataPtr(), ARLIM(f0fab.loVect()), ARLIM(f0fab.hiVect()),
                  m0.dataPtr(), ARLIM(m0.loVect()), ARLIM(m0.hiVect()),
                  f1fab.dataPtr(), ARLIM(f1fab.loVect()), ARLIM(f1fab.hiVect()),
                  m1.dataPtr(), ARLIM(m1.loVect()), ARLIM(m1.hiVect()),
                  f2fab.dataPtr(), ARLIM(f2fab.loVect()), ARLIM(f2fab.hiVect()),
                  m2.dataPtr(), ARLIM(m2.loVect()), ARLIM(m2.hiVect()),
                  f3fab.dataPtr(), ARLIM(f3fab.loVect()), ARLIM(f3fab.hiVect()),
                  m3.dataPtr(), ARLIM(m3.loVect()), ARLIM(m3.hiVect()),
                  f4fab.dataPtr(), ARLIM(f4fab.loVect()), ARLIM(f4fab.hiVect()),
                  m4.dataPtr(), ARLIM(m4.loVect()), ARLIM(m4.hiVect()),
                  f5fab.dataPtr(), ARLIM(f5fab.loVect()), ARLIM(f5fab.hiVect()),
                  m5.dataPtr(), ARLIM(m5.loVect()), ARLIM(m5.hiVect()),
                  tbx.loVect(), tbx.hiVect(), vbx.loVect(), vbx.hiVect(),
                  &nc, h[level], &redBlackFlag);
#endif
#endif
    }
}

void
ABecLaplacian::Fsmooth_jacobi (MultiFab&       solnL,
                               const MultiFab& rhsL,
                               int             level)
{
    BL_PROFILE("ABecLaplacian::Fsmooth_jacobi()");

    OrientationIter oitr;

    const FabSet& f0 = (*undrrelxr[level])[oitr()]; oitr++;
    const FabSet& f1 = (*undrrelxr[level])[oitr()]; oitr++;
    const FabSet& f2 = (*undrrelxr[level])[oitr()]; oitr++;
    const FabSet& f3 = (*undrrelxr[level])[oitr()]; oitr++;
#if (BL_SPACEDIM > 2)
    const FabSet& f4 = (*undrrelxr[level])[oitr()]; oitr++;
    const FabSet& f5 = (*undrrelxr[level])[oitr()]; oitr++;
#endif    
    const MultiFab& a = aCoefficients(level);

    D_TERM(const MultiFab& bX = bCoefficients(0,level);,
           const MultiFab& bY = bCoefficients(1,level);,
           const MultiFab& bZ = bCoefficients(2,level););

    oitr.rewind();
    const MultiMask& mm0 = maskvals[level][oitr()]; oitr++;
    const MultiMask& mm1 = maskvals[level][oitr()]; oitr++;
    const MultiMask& mm2 = maskvals[level][oitr()]; oitr++;
    const MultiMask& mm3 = maskvals[level][oitr()]; oitr++;
#if (BL_SPACEDIM > 2)
    const MultiMask& mm4 = maskvals[level][oitr()]; oitr++;
    const MultiMask& mm5 = maskvals[level][oitr()]; oitr++;
#endif

    //const int nc = solnL.nComp(); // FIXME: This LinOp only really supports single-component
    const int nc = 1;

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter solnLmfi(solnL); solnLmfi.isValid(); ++solnLmfi)
    {
	const Mask& m0 = mm0[solnLmfi];
        const Mask& m1 = mm1[solnLmfi];
        const Mask& m2 = mm2[solnLmfi];
        const Mask& m3 = mm3[solnLmfi];
#if (BL_SPACEDIM > 2)
        const Mask& m4 = mm4[solnLmfi];
        const Mask& m5 = mm5[solnLmfi];
#endif

        const Box&       vbx     = solnLmfi.validbox();
        FArrayBox&       solnfab = solnL[solnLmfi];
        const FArrayBox& rhsfab  = rhsL[solnLmfi];
        const FArrayBox& afab    = a[solnLmfi];

        D_TERM(const FArrayBox& bxfab = bX[solnLmfi];,
               const FArrayBox& byfab = bY[solnLmfi];,
               const FArrayBox& bzfab = bZ[solnLmfi];);

        const FArrayBox& f0fab = f0[solnLmfi];
        const FArrayBox& f1fab = f1[solnLmfi];
        const FArrayBox& f2fab = f2[solnLmfi];
        const FArrayBox& f3fab = f3[solnLmfi];
#if (BL_SPACEDIM > 2)
        const FArrayBox& f4fab = f4[solnLmfi];
        const FArrayBox& f5fab = f5[solnLmfi];
#endif

#if (BL_SPACEDIM == 2)
        FORT_JACOBI(solnfab.dataPtr(), ARLIM(solnfab.loVect()),ARLIM(solnfab.hiVect()),
                    rhsfab.dataPtr(), ARLIM(rhsfab.loVect()), ARLIM(rhsfab.hiVect()),
                    &alpha, &beta,
                    afab.dataPtr(), ARLIM(afab.loVect()),    ARLIM(afab.hiVect()),
                    bxfab.dataPtr(), ARLIM(bxfab.loVect()),   ARLIM(bxfab.hiVect()),
                    byfab.dataPtr(), ARLIM(byfab.loVect()),   ARLIM(byfab.hiVect()),
                    f0fab.dataPtr(), ARLIM(f0fab.loVect()),   ARLIM(f0fab.hiVect()),
                    m0.dataPtr(), ARLIM(m0.loVect()),   ARLIM(m0.hiVect()),
                    f1fab.dataPtr(), ARLIM(f1fab.loVect()),   ARLIM(f1fab.hiVect()),
                    m1.dataPtr(), ARLIM(m1.loVect()),   ARLIM(m1.hiVect()),
                    f2fab.dataPtr(), ARLIM(f2fab.loVect()),   ARLIM(f2fab.hiVect()),
                    m2.dataPtr(), ARLIM(m2.loVect()),   ARLIM(m2.hiVect()),
                    f3fab.dataPtr(), ARLIM(f3fab.loVect()),   ARLIM(f3fab.hiVect()),
                    m3.dataPtr(), ARLIM(m3.loVect()),   ARLIM(m3.hiVect()),
                    vbx.loVect(), vbx.hiVect(),
                    &nc, h[level]);
#endif

#if (BL_SPACEDIM == 3)
        FORT_JACOBI(solnfab.dataPtr(), ARLIM(solnfab.loVect()),ARLIM(solnfab.hiVect()),
                    rhsfab.dataPtr(), ARLIM(rhsfab.loVect()), ARLIM(rhsfab.hiVect()),
                    &alpha, &beta,
                    afab.dataPtr(), ARLIM(afab.loVect()), ARLIM(afab.hiVect()),
                    bxfab.dataPtr(), ARLIM(bxfab.loVect()), ARLIM(bxfab.hiVect()),
                    byfab.dataPtr(), ARLIM(byfab.loVect()), ARLIM(byfab.hiVect()),
                    bzfab.dataPtr(), ARLIM(bzfab.loVect()), ARLIM(bzfab.hiVect()),
                    f0fab.dataPtr(), ARLIM(f0fab.loVect()), ARLIM(f0fab.hiVect()),
                    m0.dataPtr(), ARLIM(m0.loVect()), ARLIM(m0.hiVect()),
                    f1fab.dataPtr(), ARLIM(f1fab.loVect()), ARLIM(f1fab.hiVect()),
                    m1.dataPtr(), ARLIM(m1.loVect()), ARLIM(m1.hiVect()),
                    f2fab.dataPtr(), ARLIM(f2fab.loVect()), ARLIM(f2fab.hiVect()),
                    m2.dataPtr(), ARLIM(m2.loVect()), ARLIM(m2.hiVect()),
                    f3fab.dataPtr(), ARLIM(f3fab.loVect()), ARLIM(f3fab.hiVect()),
                    m3.dataPtr(), ARLIM(m3.loVect()), ARLIM(m3.hiVect()),
                    f4fab.dataPtr(), ARLIM(f4fab.loVect()), ARLIM(f4fab.hiVect()),
                    m4.dataPtr(), ARLIM(m4.loVect()), ARLIM(m4.hiVect()),
                    f5fab.dataPtr(), ARLIM(f5fab.loVect()), ARLIM(f5fab.hiVect()),
                    m5.dataPtr(), ARLIM(m5.loVect()), ARLIM(m5.hiVect()),
                    vbx.loVect(), vbx.hiVect(),
                    &nc, h[level]);
#endif
    }
}

#include <fstream>
void
ABecLaplacian::Fapply (MultiFab&       y,
                       const MultiFab& x,
                       int             level)
{
  int num_comp = 1;
  int src_comp = 0;
  int dst_comp = 0;

  Fapply(y,dst_comp,x,src_comp,num_comp,level);
}

void
ABecLaplacian::Fapply (MultiFab&       y,
		       int             dst_comp,
                       const MultiFab& x,
		       int             src_comp,
		       int             num_comp,
                       int             level)
{
    BL_PROFILE("ABecLaplacian::Fapply()");

    BL_ASSERT(y.nComp()>=dst_comp+num_comp);
    BL_ASSERT(x.nComp()>=src_comp+num_comp);

    const MultiFab& a   = aCoefficients(level);

    D_TERM(const MultiFab& bX  = bCoefficients(0,level);,
           const MultiFab& bY  = bCoefficients(1,level);,
           const MultiFab& bZ  = bCoefficients(2,level););

    const bool tiling = true;

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter ymfi(y,tiling); ymfi.isValid(); ++ymfi)
    {
        const Box&       tbx  = ymfi.tilebox();
        FArrayBox&       yfab = y[ymfi];
        const FArrayBox& xfab = x[ymfi];
        const FArrayBox& afab = a[ymfi];

        D_TERM(const FArrayBox& bxfab = bX[ymfi];,
               const FArrayBox& byfab = bY[ymfi];,
               const FArrayBox& bzfab = bZ[ymfi];);

#if (BL_SPACEDIM == 2)
        FORT_ADOTX(yfab.dataPtr(dst_comp),
                   ARLIM(yfab.loVect()),ARLIM(yfab.hiVect()),
                   xfab.dataPtr(src_comp),
                   ARLIM(xfab.loVect()), ARLIM(xfab.hiVect()),
                   &alpha, &beta, afab.dataPtr(), 
                   ARLIM(afab.loVect()), ARLIM(afab.hiVect()),
                   bxfab.dataPtr(), 
                   ARLIM(bxfab.loVect()), ARLIM(bxfab.hiVect()),
                   byfab.dataPtr(), 
                   ARLIM(byfab.loVect()), ARLIM(byfab.hiVect()),
                   tbx.loVect(), tbx.hiVect(), &num_comp,
                   h[level]);
#endif
#if (BL_SPACEDIM ==3)
        FORT_ADOTX(yfab.dataPtr(dst_comp),
                   ARLIM(yfab.loVect()), ARLIM(yfab.hiVect()),
                   xfab.dataPtr(src_comp),
                   ARLIM(xfab.loVect()), ARLIM(xfab.hiVect()),
                   &alpha, &beta, afab.dataPtr(), 
                   ARLIM(afab.loVect()), ARLIM(afab.hiVect()),
                   bxfab.dataPtr(), 
                   ARLIM(bxfab.loVect()), ARLIM(bxfab.hiVect()),
                   byfab.dataPtr(), 
                   ARLIM(byfab.loVect()), ARLIM(byfab.hiVect()),
                   bzfab.dataPtr(), 
                   ARLIM(bzfab.loVect()), ARLIM(bzfab.hiVect()),
                   tbx.loVect(), tbx.hiVect(), &num_comp,
                   h[level]);
#endif
    }
}
