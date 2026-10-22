"""pytest/unittest-compatible TB-0 numerical and invariance preflight."""
import unittest
import numpy as np
import tb0

class TB0Tests(unittest.TestCase):
 def test_identity(self):
  p=tb0.preflight();self.assertTrue(p['four_distinct_atoms']);self.assertLess(max(p['fft_mismatch'].values()),1e-12)
 def test_homometric_proof(self):
  # Direct Wiener-Khinchin check: absolute FFT squared equals DFT of autocorrelation
  a=tb0.A; ac=np.correlate(a,a,'full');freq=np.fft.fft(np.r_[ac[17:],np.zeros(64-35),ac[:17]],norm=None)/64
  self.assertLess(np.max(np.abs(tb0.amp(a)**2-freq)),1e-12)
 def test_prior_and_likelihood(self):
  y=tb0.amp(tb0.B)+np.linspace(-.025,.035,64)
  self.assertLess(np.max(np.abs(tb0.posterior(y)-tb0.W)),1e-12)
 def test_grad(self):
  for alpha in (.25,.5,.8):
   x=tb0.B+np.linspace(-.04,.04,18);E,g,m,r=tb0.stats(x,alpha)
   for k in (0,3,12,17):
    h=1e-5;e=np.eye(18)[k]*h
    dg=-(tb0.stats(x+e,alpha)[0]-tb0.stats(x-e,alpha)[0])/(2*h)
    self.assertAlmostEqual(dg,g[k],places=6)
   z=x*np.sqrt(alpha);Ez,gz,mm,rr=tb0.z_stats(z,alpha)
   self.assertLess(np.linalg.norm(gz-g/np.sqrt(alpha)),1e-10)
   self.assertLess(np.linalg.norm(mm-m),1e-10)
 def test_phase_real_crop(self):
  rng=np.random.default_rng(171)
  d=tb0.hermitian_direction(rng)
  self.assertEqual(d[0],0);self.assertEqual(d[32],0)
  self.assertLess(np.max(np.abs(d[1:32]+d[-1:32:-1])),1e-12)
  u=tb0.A+np.linspace(-.1,.1,18)
  cand,b,a,imag=tb0.phase_crop(u,1.*d,tb0.new_ledger())
  self.assertLess(imag,1e-12);self.assertLess(b,1e-12)
  self.assertGreater(a,1e-5)
  self.assertEqual(len(cand),18)
 def test_aligned_zero_floor(self):
  d,n=tb0.aligned_direction(tb0.B,np.zeros(18),1e-8,1e-8)
  self.assertEqual(n,0.);self.assertTrue(np.all(d==0))
 def test_accept_monotone_reject(self):
  cfg={'alphas':[.25,.5,.8],'eta':.035,'parent_steps':1,'refine_steps':4,'s0_extra_steps':3,
       'scale':.5,'floor':1e-8,'eps':1e-8,'delta_energy':1e9,
       'measurement_gate':1e9,'label_radius':.13}
  y=tb0.amp(tb0.B);out=tb0.run_one('S3',tb0.B,y,'test_reject',cfg)
  self.assertEqual(out['num_accepted'],0)
  parent=tb0.B.copy();w=tb0.new_ledger()
  for alpha in cfg['alphas']:
   parent=tb0.refine(parent,alpha,1,cfg['eta'],w)
  self.assertAlmostEqual(out['meas_chi2'],tb0.measurement_error(parent,y))
  self.assertEqual(out['ledger']['score'],27)
  self.assertEqual(out['ledger']['fft'],27)
  self.assertEqual(out['ledger']['ifft'],6)
 def test_counter_and_pairing(self):
  for a in tb0.ARMS:
   cfg={'alphas':[.25,.5,.8],'eta':.035,'parent_steps':1,'refine_steps':4,'s0_extra_steps':3,
       'scale':.5,'floor':1e-8,'eps':1e-8,'delta_energy':.001,
       'measurement_gate':93.21685966023843,'label_radius':.13}
   q=tb0.run_one(a,tb0.B,tb0.amp(tb0.B),'test_paired',cfg)
   self.assertEqual([q['ledger'][key] for key in ('score','fft','ifft','energy','proposals')],[27,27,6,6,3])
 def test_orbit_classification(self):
  y=tb0.amp(tb0.A);gate=tb0.chi2.ppf(.99,64)
  for i in range(4):
   lab,k,d=tb0.label(tb0.MU[i],y,.13,gate)
   self.assertEqual(lab,'A' if i<2 else 'B');self.assertEqual(k,i)
  self.assertEqual(tb0.label((tb0.A+tb0.B)/2,y,.13,gate)[0],'U')
 def test_namespace_no_shared_seed(self):
  self.assertNotEqual(tb0.seed('tb0/dev/b-start/1'),tb0.seed('tb0/check/b-start/1'))
  self.assertNotEqual(tb0.seed('tb0/noise/dev/b-start/1'),tb0.seed('tb0/dev/b-start/1'))
 def test_truth_free_interface(self):
  # GT never passed to run_one, proposal, score, acceptance or label.
  import inspect
  for func in (tb0.run_one,tb0.proposal,tb0.stats,tb0.label):
   self.assertNotIn('ground_truth',inspect.signature(func).parameters)

if __name__=='__main__':unittest.main()
