#!/usr/bin/env python3
"""Independent algebraic and negative controls for the pinned llama GDN CPU ref.

Unqualified invocation runs tiny CPU tests. --integration additionally verifies
the installed llama source and genuine GGUF/capture, then derives first-token
stage hashes. No GPU, llama forward, PP2 or service invocation occurs.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"experiments"))
import llama_gdn_layer0_independent_cpu_ref as ref
import numpy as np

OPTIONS = None


class EquationTests(unittest.TestCase):
    def test_qk_l2_epsilon_scale_and_modulo_groups(self):
        # Hand-computed L2 heads [3,4] and [0,5] with EPS=4.
        q0=np.array([[3.,4.,0.,0.],[0.,0.,5.,0.]])
        k0=np.array([[0.,3.,4.,0.],[0.,0.,5.,0.]])
        v=np.arange(24,dtype=np.float64).reshape(6,4)
        conv=np.concatenate((q0.ravel(),k0.ravel(),v.ravel()))
        q,k,v2,map_h=ref.qk_normalize(
            conv,np,groups=2,heads=6,width=4,eps=4.)
        np.testing.assert_array_equal(map_h,[0,1,0,1,0,1])
        np.testing.assert_allclose(q[0],q0[0]/math.sqrt(29),rtol=0,atol=1e-15)
        np.testing.assert_allclose(k[1],k0[1]/math.sqrt(29),rtol=0,atol=1e-15)
        np.testing.assert_array_equal(q[0],q[2])
        np.testing.assert_array_equal(k[1],k[3])
        np.testing.assert_array_equal(v2,v)
        # A head//3 mutant groups [0,0,0,1,1,1] and must disagree.
        wrong=np.arange(6)//3
        self.assertGreater(np.max(np.abs(q0[wrong]-q0[map_h])),2.)
        # Source's RMS eps is eps/n, yielding L2 eps after 1/sqrt(n).
        correct=math.sqrt((q0[0]**2).sum()+4)
        wrong_eps=math.sqrt((q0[0]**2).sum()+4*4)
        self.assertGreater(abs(3/correct-3/wrong_eps),.08)

    def test_manual_delta_rule_nonzero_history_and_zero_history(self):
        q=np.array([[1.,0.]],dtype=np.float64)
        k=np.array([[1.,0.]],dtype=np.float64)
        v=np.array([[2.,4.]],dtype=np.float64)
        beta=np.array([.5])
        decay=np.array([math.log(.5)])
        previous=np.array([[[1.,3.],[0.,0.]]],dtype=np.float64)
        y,state,d=ref.delta_step(q,k,v,beta,decay,previous,np)
        # Prediction [.5,1.5]; innovation .5 * [1.5,2.5].
        np.testing.assert_allclose(d,[[.75,1.25]],atol=1e-15,rtol=0)
        np.testing.assert_allclose(state,[[[1.25,2.75],[0,0]]],
                                   atol=1e-15,rtol=0)
        np.testing.assert_allclose(y,[[1.25/math.sqrt(2),2.75/math.sqrt(2)]],
                                   atol=1e-15,rtol=0)
        wrong_no_scale=np.array([[1.25,2.75]])
        self.assertGreater(float(np.max(abs(y-wrong_no_scale))),.4)
        wrong_no_prediction=np.array([[[1.5,3.5],[0,0]]])
        self.assertGreater(float(np.max(abs(state-wrong_no_prediction))),.2)
        zero=np.zeros_like(previous)
        y0,s0,d0=ref.delta_step(q,k,v,beta,decay,zero,np)
        y1,s1,d1=ref.delta_step(q,k,v,beta,np.array([-20.]),zero,np)
        np.testing.assert_array_equal(y0,y1)
        np.testing.assert_array_equal(s0,s1)
        np.testing.assert_array_equal(d0,d1)
        np.testing.assert_allclose(y0,[[1/math.sqrt(2),2/math.sqrt(2)]],
                                   atol=1e-15,rtol=0)
        # Zero-history cannot test gate; nonzero-history can.
        nonzero_other,_,_=ref.delta_step(
            q,k,v,beta,np.array([-20.]),previous,np)
        self.assertGreater(float(np.max(abs(nonzero_other-y))),.2)

    def test_beta_sigmoid_softplus_and_negative_stored_a(self):
        raw=np.array([0.,math.log(3.)])
        alpha=np.array([0.,1.])
        dt=np.array([0.,-1.])
        neg_a=np.array([-2.,-3.])
        beta,g=ref.source_gate(raw,alpha,dt,neg_a,np)
        np.testing.assert_allclose(beta,[.5,.75],atol=1e-15,rtol=0)
        np.testing.assert_allclose(g,[-2*math.log(2),-3*math.log(2)],
                                   atol=1e-15,rtol=0)
        self.assertGreater(float(np.max(abs(beta-raw))),.4)
        with self.assertRaisesRegex(ref.Blocked,"negative exp"):
            ref.source_gate(raw,alpha,dt,np.array([2.,-3.]),np)

    def test_gated_rms_gated_silu_and_mutant_rejection(self):
        y=np.array([[3.,4.]],dtype=np.float64)
        z=np.array([[0.,1.]])
        weight=np.array([2.,3.])
        result=ref.gated_rms(y,z,weight,np,eps=0)
        expected=4/math.sqrt(12.5)*3/(1+math.exp(-1))
        np.testing.assert_allclose(result,[[0.,expected]],atol=1e-15,rtol=0)
        self.assertGreater(abs(result[0,1]-4/math.sqrt(25)*3/(1+math.exp(-1))),.7)
        np.testing.assert_array_equal(ref.gated_rms(
            y,np.zeros_like(z),weight,np),[[0.,0.]])

    def test_q8_per32_ties_zero_and_sign(self):
        a=np.zeros(64,dtype=np.float32)
        a[0]=127
        a[1]=1.5
        a[2]=-1.5
        a[3]=.49
        a[32]=0
        x=ref.q8_per32(a,np)
        np.testing.assert_array_equal(x[:4],np.array([127.,2.,-2.,0.]))
        np.testing.assert_array_equal(x[32:],np.zeros(32))
        with self.assertRaisesRegex(ref.Blocked,"Q8_1"):
            ref.q8_per32(np.array([1.,2.]),np)

    def test_gguf_rms_epsilon_metadata_positive_and_mutants(self):
        key=b"qwen35.attention.layer_norm_rms_epsilon"
        marker=struct.pack("<Q",len(key))+key+struct.pack("<I",6)
        class Metadata:
            def __init__(self,data):
                self.mm=data
                self.data_offset=len(data)
        good=Metadata(bytes(24)+marker+struct.pack("<f",1e-6))
        self.assertAlmostEqual(ref.gguf_rms_epsilon(good),1e-6,places=12)
        with self.assertRaisesRegex(ref.Blocked,"missing GGUF"):
            ref.gguf_rms_epsilon(Metadata(bytes(24)+b"wrong"+struct.pack("<f",1e-6)))
        with self.assertRaisesRegex(ref.Blocked,"does not match"):
            ref.gguf_rms_epsilon(Metadata(bytes(24)+marker+struct.pack("<f",1e-4)))

    def test_finite_bad_geometry_and_source_mutants(self):
        with self.assertRaisesRegex(ref.Blocked,"head ratio"):
            ref.qk_normalize(np.ones(20),np,groups=3,heads=5,width=1)
        with self.assertRaisesRegex(ref.Blocked,"delta-net"):
            ref.delta_step(np.ones((1,2)),np.ones((2,2)),
                           np.ones((1,2)),np.ones(1),
                           np.zeros(1),np.zeros((1,2,2)),np)
        with self.assertRaisesRegex(ref.Blocked,"beta or state decay"):
            ref.delta_step(np.ones((1,2)),np.ones((1,2)),
                           np.ones((1,2)),np.array([1.1]),
                           np.zeros(1),np.zeros((1,2,2)),np)
        for key, fragments in ref.SOURCE_CONTRACT.items():
            good="\n".join(fragments)
            ref.check_contract_text(key,good)
            with self.subTest(source=key):
                with self.assertRaisesRegex(ref.Blocked,"contract missing"):
                    ref.check_contract_text(key,good.replace(fragments[0],
                                                            "MUTATED_FORMULA"))
        with self.assertRaisesRegex(ref.Blocked,"missing installed llama"):
            ref.verify_installed_llama(Path("/not/a/real/llama-dir"))


class RealGGUFTests(unittest.TestCase):
    def test_real_q8_input_against_existing_cpu_quantizer(self):
        if OPTIONS is None:
            self.skipTest("requires authenticated real capture")
        sys.path.insert(0,str(OPTIONS.source_root/"tests"))
        import reliability_layer0_conv_q8_cpu as previous
        x=np.fromfile(OPTIONS.manifest.parent/"attn_norm-0.f32",dtype="<f4")
        old=previous.quant_q8_1_per32(x,np)
        new=ref.q8_per32(x,np)
        # The two implementations may encode signed zero differently.
        np.testing.assert_array_equal(new,old)
        bit_diffs=np.count_nonzero(new.view(np.uint32)!=old.view(np.uint32))
        print("AUTH_CAPTURE_Q8_FLOATS_EQUAL; bit-differing zero signs:",bit_diffs)
        self.assertTrue(np.array_equal(new,old))

    def test_real_first_token_cpu(self):
        if OPTIONS is None:
            self.skipTest("provide --manifest --gguf --source-root --llama-root --out")
        result=ref.run(OPTIONS.manifest,OPTIONS.gguf,OPTIONS.source_root,
                       OPTIONS.llama_root,OPTIONS.out)
        self.assertEqual(result["schema"],ref.SCHEMA)
        self.assertEqual(result["token"],6205)
        self.assertEqual(result["absolute_position"],0)
        self.assertEqual(result["status"],
                         "CPU_EQUATION_ONLY_NO_AUTHENTIC_LLAMA_POSTCONV_REFERENCE")
        stages=result["stages_cpu_derived_not_llama"]
        self.assertEqual(stages["beta_sigmoid"]["shape"],[48])
        self.assertEqual(stages["gdn_output"]["shape"],[48,128])
        self.assertEqual(stages["gated_rms_silu"]["shape"],[48,128])
        self.assertEqual(len(result["authentic_llama_tensors"]),3)
        self.assertFalse("llama_gdn_output" in result)
        self.assertEqual(json.loads(OPTIONS.out.read_text())["schema"],ref.SCHEMA)

    def test_corrupted_real_manifest_blocked_before_gguf_read(self):
        if OPTIONS is None:
            self.skipTest("requires authenticated real capture")
        with tempfile.TemporaryDirectory() as directory:
            corrupt=Path(directory)/"manifest.json"
            content=OPTIONS.manifest.read_bytes()
            corrupt.write_bytes(content.replace(b'"prompt_ids":[6205]',
                                                b'"prompt_ids":[6206]'))
            self.assertNotEqual(ref.sha256(corrupt),ref.MANIFEST_SHA)
            with self.assertRaisesRegex(ref.Blocked,"manifest SHA"):
                ref.run(corrupt,OPTIONS.gguf,OPTIONS.source_root,
                        OPTIONS.llama_root,Path(directory)/"fake.json")


def main():
    global OPTIONS
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("manifest","gguf","source-root","llama-root","out"):
        parser.add_argument("--"+key,type=Path)
    args=parser.parse_args()
    attrs=("manifest","gguf","source_root","llama_root","out")
    if any(getattr(args,key) for key in attrs):
        parser.error("real integration requires all five --manifest/--gguf/"
                     "--source-root/--llama-root/--out paths") if not all(
                         getattr(args,key) for key in attrs) else None
        OPTIONS=args
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__=="__main__":
    sys.exit(main())
