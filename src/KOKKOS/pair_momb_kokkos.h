/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(momb/kk,PairMombKokkos<LMPDeviceType>)
PairStyle(momb/kk/device,PairMombKokkos<LMPDeviceType>)
PairStyle(momb/kk/host,PairMombKokkos<LMPHostType>)

#else

#ifndef LMP_PAIR_MOMB_H
#define LMP_PAIR_MOMB_H

#include "pair_kokkos.h"
#include "pair_momb.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairMombKokkos : public PairMomb {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF|N2};
  enum {COUL_FLAG=0};
  typedef DeviceType device_type;
  PairMombKokkos(class LAMMPS *);
  virtual ~PairMombKokkos();

  void compute(int, int);

  void settings(int, char **);
  void init_style();
  double init_one(int, int);

  struct params_momb{
    KOKKOS_INLINE_FUNCTION
    params_momb(){cutsq=0,d0=0;alpha=0;r0=0;c=0;rr=0;offset=0;}
    KOKKOS_INLINE_FUNCTION
    params_momb(int i){cutsq=0,d0=0;alpha=0;r0=0;c=0;rr=0;offset=0;}
    F_FLOAT cutsq,d0,alpha,r0,c,rr,offset;
  };

 protected:
  void cleanup_copy();

  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_fpair(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const;

  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_evdwl(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const;

  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_ecoul(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const {
    return 0;
  }


  Kokkos::DualView<params_momb**,Kokkos::LayoutRight,DeviceType> k_params;
  typename Kokkos::DualView<params_momb**,Kokkos::LayoutRight,DeviceType>::t_dev_const_um params;
  params_momb m_params[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1]; 
  F_FLOAT m_cutsq[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];
  typename ArrayTypes<DeviceType>::t_x_array_randomread x;
  typename ArrayTypes<DeviceType>::t_x_array c_x;
  typename ArrayTypes<DeviceType>::t_f_array f;
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename ArrayTypes<DeviceType>::t_efloat_1d d_eatom;
  typename ArrayTypes<DeviceType>::t_virial_array d_vatom;
  typename ArrayTypes<DeviceType>::t_tagint_1d tag;

  int newton_pair;
  double special_lj[4];

  typename ArrayTypes<DeviceType>::tdual_ffloat_2d k_cutsq;
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq;


  int neighflag;
  int nlocal,nall,eflag,vflag;

  void allocate();
  friend class PairComputeFunctor<PairMombKokkos,FULL,true>;
  friend class PairComputeFunctor<PairMombKokkos,HALF,true>;
  friend class PairComputeFunctor<PairMombKokkos,HALFTHREAD,true>;
  friend class PairComputeFunctor<PairMombKokkos,N2,true>;
  friend class PairComputeFunctor<PairMombKokkos,FULL,false>;
  friend class PairComputeFunctor<PairMombKokkos,HALF,false>;
  friend class PairComputeFunctor<PairMombKokkos,HALFTHREAD,false>;
  friend class PairComputeFunctor<PairMombKokkos,N2,false>;
  friend EV_FLOAT pair_compute_neighlist<PairMombKokkos,FULL,void>(PairMombKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairMombKokkos,HALF,void>(PairMombKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairMombKokkos,HALFTHREAD,void>(PairMombKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairMombKokkos,N2,void>(PairMombKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute<PairMombKokkos,void>(PairMombKokkos*,NeighListKokkos<DeviceType>*);
  friend void pair_virial_fdotr_compute<PairMombKokkos>(PairMombKokkos*);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Cannot use Kokkos pair style with rRESPA inner/middle

Self-explanatory.

E: Cannot use chosen neighbor list style with momb/kk

That style is not supported by Kokkos.

*/
