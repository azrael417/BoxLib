
#include <winstd.H>
#include <algorithm>
#include <cfloat>
#include <iostream>
#include <iomanip>
#include <map>
#include <limits>

#include <BLassert.H>
#include <iMultiFab.H>
#include <ParallelDescriptor.H>
#include <BLProfiler.H>
#include <ParmParse.H>
#include <PArray.H>

namespace
{
    bool initialized = false;
}

void
iMultiFab::Add (iMultiFab&       dst,
	       const iMultiFab& src,
	       int             srccomp,
	       int             dstcomp,
	       int             numcomp,
	       int             nghost)
{
    BL_ASSERT(dst.boxArray() == src.boxArray());
    BL_ASSERT(dst.distributionMap == src.distributionMap);
    BL_ASSERT(dst.nGrow() >= nghost && src.nGrow() >= nghost);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(dst,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);

        if (bx.ok())
            dst[mfi].plus(src[mfi], bx, bx, srccomp, dstcomp, numcomp);
    }
}

void
iMultiFab::Copy (iMultiFab&       dst,
                const iMultiFab& src,
                int             srccomp,
                int             dstcomp,
                int             numcomp,
                int             nghost)
{
    BL_ASSERT(dst.boxArray() == src.boxArray());
    BL_ASSERT(dst.distributionMap == src.distributionMap);
    BL_ASSERT(dst.nGrow() >= nghost && src.nGrow() >= nghost);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(dst,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);

        if (bx.ok())
            dst[mfi].copy(src[mfi], bx, srccomp, bx, dstcomp, numcomp);
    }
}

void
iMultiFab::Subtract (iMultiFab&       dst,
		    const iMultiFab& src,
		    int             srccomp,
		    int             dstcomp,
		    int             numcomp,
		    int             nghost)
{
    BL_ASSERT(dst.boxArray() == src.boxArray());
    BL_ASSERT(dst.distributionMap == src.distributionMap);
    BL_ASSERT(dst.nGrow() >= nghost && src.nGrow() >= nghost);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(dst,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);

        if (bx.ok())
            dst[mfi].minus(src[mfi], bx, bx, srccomp, dstcomp, numcomp);
    }
}

void
iMultiFab::Multiply (iMultiFab&       dst,
		    const iMultiFab& src,
		    int             srccomp,
		    int             dstcomp,
		    int             numcomp,
		    int             nghost)
{
    BL_ASSERT(dst.boxArray() == src.boxArray());
    BL_ASSERT(dst.distributionMap == src.distributionMap);
    BL_ASSERT(dst.nGrow() >= nghost && src.nGrow() >= nghost);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(dst,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);

        if (bx.ok())
            dst[mfi].mult(src[mfi], bx, bx, srccomp, dstcomp, numcomp);
    }
}

void
iMultiFab::Divide (iMultiFab&       dst,
		  const iMultiFab& src,
		  int             srccomp,
		  int             dstcomp,
		  int             numcomp,
		  int             nghost)
{
    BL_ASSERT(dst.boxArray() == src.boxArray());
    BL_ASSERT(dst.distributionMap == src.distributionMap);
    BL_ASSERT(dst.nGrow() >= nghost && src.nGrow() >= nghost);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(dst,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);

        if (bx.ok())
            dst[mfi].divide(src[mfi], bx, bx, srccomp, dstcomp, numcomp);
    }
}

void
iMultiFab::plus (int val,
                 int  nghost)
{
    plus(val,0,n_comp,nghost);
}

void
iMultiFab::plus (int       val,
                 const Box& region,
                 int        nghost)
{
    plus(val,region,0,n_comp,nghost);
}

void
iMultiFab::mult (int val,
                 int  nghost)
{
    mult(val,0,n_comp,nghost);
}

void
iMultiFab::mult (int       val,
                 const Box& region,
                 int        nghost)
{
    mult(val,region,0,n_comp,nghost);
}

void
iMultiFab::negate (int nghost)
{
    negate(0,n_comp,nghost);
}

void
iMultiFab::negate (const Box& region,
                  int        nghost)
{
    negate(region,0,n_comp,nghost);
}

void
iMultiFab::Initialize ()
{
    if (initialized) return;

    BoxLib::ExecOnFinalize(iMultiFab::Finalize);

    initialized = true;
}

void
iMultiFab::Finalize ()
{
    initialized = false;
}

iMultiFab::iMultiFab ()
{
    Initialize();
}

iMultiFab::iMultiFab (const BoxArray& bxs,
                    int             ncomp,
                    int             ngrow,
                    FabAlloc        alloc)
    :
    FabArray<IArrayBox>(bxs,ncomp,ngrow,alloc)
{
    Initialize();
}

iMultiFab::iMultiFab (const BoxArray&            bxs,
                      int                        ncomp,
                      int                        ngrow,
                      const DistributionMapping& dm,
                      FabAlloc                   alloc)
    :
    FabArray<IArrayBox>(bxs,ncomp,ngrow,dm,alloc)
{
    Initialize();
}

void
iMultiFab::operator= (const int& r)
{
    setVal(r);
}

void
iMultiFab::define (const BoxArray& bxs,
                  int             nvar,
                  int             ngrow,
                  FabAlloc        alloc)
{
    this->FabArray<IArrayBox>::define(bxs,nvar,ngrow,alloc);
}

void
iMultiFab::define (const BoxArray&            bxs,
                  int                        nvar,
                  int                        ngrow,
                  const DistributionMapping& dm,
                  FabAlloc                   alloc)
{
    this->FabArray<IArrayBox>::define(bxs,nvar,ngrow,dm,alloc);
}


const IArrayBox&
iMultiFab::operator[] (int K) const
{
    BL_ASSERT(defined(K));

    const IArrayBox& fab = this->FabArray<IArrayBox>::get(K);

    return fab;
}

IArrayBox&
iMultiFab::operator[] (int K)
{
    BL_ASSERT(defined(K));

    IArrayBox& fab = this->FabArray<IArrayBox>::get(K);

    return fab;
}

int
iMultiFab::min (int comp,
		int nghost,
		bool local) const
{
    BL_ASSERT(nghost >= 0 && nghost <= n_grow);

    int mn = std::numeric_limits<int>::max();

#ifdef _OPENMP
#pragma omp parallel reduction(min:mn)
#endif
    for (MFIter mfi(*this,true); mfi.isValid(); ++mfi)
    {
	const Box& bx = mfi.growntilebox(nghost);
	mn = std::min(mn,get(mfi).min(bx,comp));
    }

    if (!local)
	ParallelDescriptor::ReduceIntMin(mn, this->color());

    return mn;
}

int
iMultiFab::min (const Box& region,
                int        comp,
                int        nghost,
		bool       local) const
{
    BL_ASSERT(nghost >= 0 && nghost <= n_grow);

    int mn = std::numeric_limits<int>::max();

#ifdef _OPENMP
#pragma omp parallel reduction(min:mn)
#endif
    for ( MFIter mfi(*this,true); mfi.isValid(); ++mfi)
    {
	const Box& b = mfi.growntilebox(nghost) & region;
	
	if (b.ok())
	    mn = std::min(mn, get(mfi).min(b,comp));
    }

    if (!local)
	ParallelDescriptor::ReduceIntMin(mn, this->color());

    return mn;
}

int
iMultiFab::max (int comp,
		int nghost,
		bool local) const
{
    BL_ASSERT(nghost >= 0 && nghost <= n_grow);

    int mx = -std::numeric_limits<int>::max();

#ifdef _OPENMP
#pragma omp parallel reduction(max:mx)
#endif
    for ( MFIter mfi(*this,true); mfi.isValid(); ++mfi)
    {
	mx = std::max(mx, get(mfi).max(mfi.growntilebox(nghost),comp));
    }

    if (!local)
	ParallelDescriptor::ReduceIntMax(mx, this->color());

    return mx;
}

int
iMultiFab::max (const Box& region,
		int        comp,
		int        nghost,
		bool       local) const
{
    BL_ASSERT(nghost >= 0 && nghost <= n_grow);

    int mx = -std::numeric_limits<int>::max();

#ifdef _OPENMP
#pragma omp parallel reduction(max:mx)
#endif
    for (MFIter mfi(*this,true); mfi.isValid(); ++mfi)
    {
	const Box& b = mfi.growntilebox(nghost) & region;
	
	if (b.ok())
	    mx = std::max(mx, get(mfi).max(b,comp));
    }

    if (!local)
	ParallelDescriptor::ReduceIntMax(mx, this->color());

    return mx;
}

IntVect
iMultiFab::minIndex (int comp,
                    int nghost) const
{
    BL_ASSERT(nghost >= 0 && nghost <= n_grow);
    BL_ASSERT(this->color() == ParallelDescriptor::DefaultColor());

    IntVect loc;

    int mn = std::numeric_limits<int>::max();

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
	IntVect priv_loc;
	int priv_mn = std::numeric_limits<int>::max();
	
	for (MFIter mfi(*this); mfi.isValid(); ++mfi)
	{
	    const Box& box = BoxLib::grow(mfi.validbox(),nghost);
	    const int  lmn = get(mfi).min(box,comp);
	    
	    if (lmn < priv_mn)
	    {
		priv_mn  = lmn;
		priv_loc = get(mfi).minIndex(box,comp);
	    }
	}

#ifdef _OPENMP
#pragma omp critical (imultifab_minindex)
#endif
	{
	    if (priv_mn < mn) {
		mn = priv_mn;
		loc = priv_loc;
	    }
	}
    }

    const int NProcs = ParallelDescriptor::NProcs();

    if (NProcs > 1)
    {
        Array<int> mns(1);
        Array<int>  locs(1);

        if (ParallelDescriptor::IOProcessor())
        {
            mns.resize(NProcs);
            locs.resize(NProcs*BL_SPACEDIM);
        }

        const int IOProc = ParallelDescriptor::IOProcessorNumber();

        ParallelDescriptor::Gather(&mn, 1, mns.dataPtr(), 1, IOProc);

        BL_ASSERT(sizeof(IntVect) == sizeof(int)*BL_SPACEDIM);

        ParallelDescriptor::Gather(loc.getVect(), BL_SPACEDIM, locs.dataPtr(), BL_SPACEDIM, IOProc);

        if (ParallelDescriptor::IOProcessor())
        {
            mn  = mns[0];
            loc = IntVect(D_DECL(locs[0],locs[1],locs[2]));

            for (int i = 1; i < NProcs; i++)
            {
                if (mns[i] < mn)
                {
                    mn = mns[i];

                    const int j = BL_SPACEDIM * i;

                    loc = IntVect(D_DECL(locs[j+0],locs[j+1],locs[j+2]));
                }
            }
        }

        ParallelDescriptor::Bcast(const_cast<int*>(loc.getVect()), BL_SPACEDIM, IOProc);
    }

    return loc;
}

IntVect
iMultiFab::maxIndex (int comp,
                    int nghost) const
{
    BL_ASSERT(nghost >= 0 && nghost <= n_grow);
    BL_ASSERT(this->color() == ParallelDescriptor::DefaultColor());

    IntVect loc;

    int mx = -std::numeric_limits<int>::max();

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
	IntVect priv_loc;
	int priv_mx = -std::numeric_limits<int>::max();

	for (MFIter mfi(*this); mfi.isValid(); ++mfi)
	{
	    const Box& box = BoxLib::grow(mfi.validbox(),nghost);
	    const int  lmx = get(mfi).max(box,comp);
	    
	    if (lmx > priv_mx)
	    {
		priv_mx  = lmx;
		priv_loc = get(mfi).maxIndex(box,comp);
	    }
	}

#ifdef _OPENMP
#pragma omp critical (imultifab_maxindex)
#endif
	{
	    if (priv_mx > mx) {
		mx = priv_mx;
		loc = priv_loc;
	    }
	}
    }

    const int NProcs = ParallelDescriptor::NProcs();

    if (NProcs > 1)
    {
        Array<int> mxs(1);
        Array<int>  locs(1);

        if (ParallelDescriptor::IOProcessor())
        {
            mxs.resize(NProcs);
            locs.resize(NProcs*BL_SPACEDIM);
        }

        const int IOProc = ParallelDescriptor::IOProcessorNumber();

        ParallelDescriptor::Gather(&mx, 1, mxs.dataPtr(), 1, IOProc);

        BL_ASSERT(sizeof(IntVect) == sizeof(int)*BL_SPACEDIM);

        ParallelDescriptor::Gather(loc.getVect(), BL_SPACEDIM, locs.dataPtr(), BL_SPACEDIM, IOProc);

        if (ParallelDescriptor::IOProcessor())
        {
            mx  = mxs[0];
            loc = IntVect(D_DECL(locs[0],locs[1],locs[2]));

            for (int i = 1; i < NProcs; i++)
            {
                if (mxs[i] > mx)
                {
                    mx = mxs[i];

                    const int j = BL_SPACEDIM * i;

                    loc = IntVect(D_DECL(locs[j+0],locs[j+1],locs[j+2]));
                }
            }
        }

        ParallelDescriptor::Bcast(const_cast<int*>(loc.getVect()), BL_SPACEDIM, IOProc);
    }

    return loc;
}

int
iMultiFab::norm0 (int comp, const BoxArray& ba, int nghost, bool local) const
{
    int nm0 = -std::numeric_limits<int>::max();

#ifdef _OPENMP
#pragma omp parallel reduction(max:nm0)
#endif
    {
	std::vector< std::pair<int,Box> > isects;

	for (MFIter mfi(*this); mfi.isValid(); ++mfi)
	{
	    ba.intersections(BoxLib::grow(mfi.validbox(),nghost),isects);
	    
	    for (int i = 0, N = isects.size(); i < N; i++)
	    {
		nm0 = std::max(nm0, get(mfi).norm(isects[i].second, 0, comp, 1));
	    }
	}
    }
 
    if (!local)
	ParallelDescriptor::ReduceIntMax(nm0, this->color());
 
    return nm0;
}

int
iMultiFab::norm0 (int comp, int nghost, bool local) const
{
    int nm0 = -std::numeric_limits<int>::max();

#ifdef _OPENMP
#pragma omp parallel reduction(max:nm0)
#endif
    for (MFIter mfi(*this,true); mfi.isValid(); ++mfi)
    {
	nm0 = std::max(nm0, get(mfi).norm(mfi.growntilebox(nghost), 0, comp, 1));
    }

    if (!local)
	ParallelDescriptor::ReduceIntMax(nm0, this->color());

    return nm0;
}

int
iMultiFab::norm2 (int comp) const
{
    int nm2 = 0;

#ifdef _OPENMP
#pragma omp parallel reduction(+:nm2)
#endif
    for (MFIter mfi(*this,true); mfi.isValid(); ++mfi)
    {
        const int nm_grid = get(mfi).norm(mfi.tilebox(), 2, comp, 1);

        nm2 += nm_grid*nm_grid;
    }

    ParallelDescriptor::ReduceIntSum(nm2, this->color());

    nm2 = std::sqrt(double(nm2));

    return nm2;
}
 
int
iMultiFab::norm1 (int comp, int ngrow, bool local) const
{
    int nm1 = 0.e0;

#ifdef _OPENMP
#pragma omp parallel reduction(+:nm1)
#endif
    for (MFIter mfi(*this); mfi.isValid(); ++mfi)
    {
        nm1 += get(mfi).norm(mfi.growntilebox(ngrow), 1, comp, 1);
    }

    if (!local)
	ParallelDescriptor::ReduceIntSum(nm1, this->color());

    return nm1;
}


void
iMultiFab::minus (const iMultiFab& mf,
                 int             strt_comp,
                 int             num_comp,
                 int             nghost)
{
    BL_ASSERT(boxarray == mf.boxarray);
    BL_ASSERT(strt_comp >= 0);
    BL_ASSERT(num_comp > 0);
    BL_ASSERT(strt_comp + num_comp - 1 < n_comp && strt_comp + num_comp - 1 < mf.n_comp);
    BL_ASSERT(nghost <= n_grow && nghost <= mf.n_grow);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(*this,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);

        get(mfi).minus(mf[mfi], bx, strt_comp, strt_comp, num_comp);
    }
}

void
iMultiFab::divide (const iMultiFab& mf,
		  int             strt_comp,
		  int             num_comp,
		  int             nghost)
{
    BL_ASSERT(boxarray == mf.boxarray);
    BL_ASSERT(strt_comp >= 0);
    BL_ASSERT(num_comp > 0);
    BL_ASSERT(strt_comp + num_comp - 1 < n_comp && strt_comp + num_comp - 1 < mf.n_comp);
    BL_ASSERT(nghost <= n_grow && nghost <= mf.n_grow);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(*this,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);

        get(mfi).divide(mf[mfi], bx, strt_comp, strt_comp, num_comp);
    }
}

void
iMultiFab::plus (int val,
                 int  comp,
                 int  num_comp,
                 int  nghost)
{
    BL_ASSERT(nghost >= 0 && nghost <= n_grow);
    BL_ASSERT(comp+num_comp <= n_comp);
    BL_ASSERT(num_comp > 0);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(*this,true); mfi.isValid(); ++mfi)
    {
        get(mfi).plus(val, mfi.growntilebox(nghost),comp,num_comp);
    }
}

void
iMultiFab::plus (int       val,
                 const Box& region,
                 int        comp,
                 int        num_comp,
                 int        nghost)
{
    BL_ASSERT(nghost >= 0 && nghost <= n_grow);
    BL_ASSERT(comp+num_comp <= n_comp);
    BL_ASSERT(num_comp > 0);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(*this,true); mfi.isValid(); ++mfi)
    {
        const Box& b = mfi.growntilebox(nghost) & region;

        if (b.ok())
            get(mfi).plus(val,b,comp,num_comp);
    }
}

void
iMultiFab::plus (const iMultiFab& mf,
                int             strt_comp,
                int             num_comp,
                int             nghost)
{
    BL_ASSERT(boxarray == mf.boxarray);
    BL_ASSERT(strt_comp >= 0);
    BL_ASSERT(num_comp > 0);
    BL_ASSERT(strt_comp + num_comp - 1 < n_comp && strt_comp + num_comp - 1 < mf.n_comp);
    BL_ASSERT(nghost <= n_grow && nghost <= mf.n_grow);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(*this,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);

        get(mfi).plus(mf[mfi], bx, strt_comp, strt_comp, num_comp);
    }
}

void
iMultiFab::mult (int val,
                 int  comp,
                 int  num_comp,
                 int  nghost)
{
    BL_ASSERT(nghost >= 0 && nghost <= n_grow);
    BL_ASSERT(comp+num_comp <= n_comp);
    BL_ASSERT(num_comp > 0);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(*this,true); mfi.isValid(); ++mfi)
    {
        get(mfi).mult(val, mfi.growntilebox(nghost),comp,num_comp);
    }
}

void
iMultiFab::mult (int       val,
                 const Box& region,
                 int        comp,
                 int        num_comp,
                 int        nghost)
{
    BL_ASSERT(nghost >= 0 && nghost <= n_grow);
    BL_ASSERT(comp+num_comp <= n_comp);
    BL_ASSERT(num_comp > 0);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(*this,true); mfi.isValid(); ++mfi)
    {
        const Box& b = mfi.growntilebox(nghost) & region;

        if (b.ok())
            get(mfi).mult(val, b, comp, num_comp);
    }
}

void
iMultiFab::negate (int comp,
                  int num_comp,
                  int nghost)
{
    BL_ASSERT(nghost >= 0 && nghost <= n_grow);
    BL_ASSERT(comp+num_comp <= n_comp);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(*this,true); mfi.isValid(); ++mfi)
    {
        get(mfi).negate(mfi.growntilebox(nghost),comp,num_comp);
    }
}

void
iMultiFab::negate (const Box& region,
                  int        comp,
                  int        num_comp,
                  int        nghost)
{
    BL_ASSERT(nghost >= 0 && nghost <= n_grow);
    BL_ASSERT(comp+num_comp <= n_comp);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(*this,true); mfi.isValid(); ++mfi)
    {
        const Box& b = mfi.growntilebox(nghost) & region;

        if (b.ok())
            get(mfi).negate(b,comp,num_comp);
    }
}
