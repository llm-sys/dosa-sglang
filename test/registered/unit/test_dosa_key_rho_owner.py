"""CPU checks of owner/current-scale routing, without CUDA or numerical samples."""
import ast,importlib.util,unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[3]/'python/sglang/srt/layers/attention/dsa/dosa_runtime'
spec=importlib.util.spec_from_file_location('key_rho_cpu_test',ROOT/'key_rho_runtime.py')
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)

class Auxiliary:
    def __init__(self):self.calls=[]
    def key_rho_contract(self):return runtime.CONTRACT
    def rms_qualified(self,k,scale,rho):self.calls.append(('flat',k,scale['epoch'],rho))
    def rms_live_nk_qualified(self,k,scale,rho,live,sms,seed):
        self.calls.append(('live',k,scale['epoch'],rho,live,sms,seed))

def owner():
    caller=NS(key_rho_contract=lambda:runtime.CONTRACT)
    consumer=NS(required_key_rho_contract=lambda:runtime.CONTRACT)
    sample=object();exact=object()
    return NS(hot_graph=None,last_done=None,modules={'caller':caller,'h16':consumer,'sample':sample},
        gather_adapter=NS(module=caller),rt=NS(core=sample,live_exact=exact,
        _required_key_rho_contract=runtime.CONTRACT,key_rho_contract=None,_qualified_rms=None,
        rho_from_gather=True,rho_source_signature=object()))

class KeyRhoOwnerTest(unittest.TestCase):
    def test_current_scale_is_forwarded_for_flat_and_live(self):
        o=owner();aux=Auxiliary();sample=o.rt.core;exact=o.rt.live_exact
        runtime.install_if_required(o,aux)
        scale={'epoch':1};k=object();rho=object();live=object()
        o.rt._qualified_rms(k,scale,rho,None,152,12288)
        scale['epoch']=2
        o.rt._qualified_rms(k,scale,rho,live,152,12288)
        self.assertEqual(aux.calls,[('flat',k,1,rho),('live',k,2,rho,live,152,12288)])
        self.assertIs(o.rt.core,sample);self.assertIs(o.rt.live_exact,exact)

    def test_two_owners_do_not_share_mutable_binding(self):
        a,b=owner(),owner();aux=Auxiliary()
        runtime.install_if_required(a,aux);runtime.install_if_required(b,aux)
        self.assertIsNot(a.rt._qualified_rms,b.rt._qualified_rms)
        self.assertIs(a.rt._qualified_rms.owner,a);self.assertIs(b.rt._qualified_rms.owner,b)
        old=a.rt._qualified_rms;runtime.install_if_required(a,aux)
        self.assertIs(a.rt._qualified_rms,old)
        with self.assertRaisesRegex(RuntimeError,'replace'):
            runtime.install_if_required(a,Auxiliary())

    def test_unqualified_or_replaced_producer_is_rejected(self):
        o=owner();o.modules['caller'].key_rho_contract=lambda:'old'
        with self.assertRaisesRegex(RuntimeError,'Unqualified'):
            runtime.install_if_required(o,Auxiliary())
        o=owner();o.gather_adapter.module=object()
        with self.assertRaisesRegex(RuntimeError,'identity'):
            runtime.install_if_required(o,Auxiliary())

    def test_ordinary_live_contracts_and_install_order(self):
        o=owner();components={'h16':o.modules['h16']}
        with self.assertRaisesRegex(RuntimeError,'Install'):
            runtime.validate_live_contract(o,components)
        runtime.install_if_required(o,Auxiliary())
        runtime.validate_live_contract(o,components)
        with self.assertRaisesRegex(RuntimeError,'agree'):
            runtime.validate_live_contract(o,{'h16':object()})

    def test_old_h16_does_not_load_auxiliary(self):
        o=NS(modules={'h16':object()})
        self.assertIs(runtime.install_if_required(o),o)

    def test_runtime_guard_and_explicit_scale_source(self):
        tree=ast.parse((ROOT/'few_core.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='FusedRuntime')
        body=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='__call__')
        guard=body.body[1]
        args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self')],kwonlyargs=[],kw_defaults=[],defaults=[])
        fn=ast.FunctionDef(name='check',args=args,body=[guard],decorator_list=[])
        ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),
                           'actual_key_guard','exec'),ns)
        o=owner()
        with self.assertRaisesRegex(RuntimeError,'current-scale'):
            ns['check'](o.rt)
        runtime.install_if_required(o,Auxiliary());ns['check'](o.rt)
        calls=[n for n in ast.walk(body)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)
               and n.func.attr=='_qualified_rms']
        self.assertEqual(len(calls),1)
        self.assertIsInstance(calls[0].args[1],ast.Name)
        self.assertEqual(calls[0].args[1].id,'scale')

if __name__=='__main__':unittest.main()
