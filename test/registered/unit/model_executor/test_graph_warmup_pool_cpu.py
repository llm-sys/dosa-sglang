"""CPU control/lifetime tests for the real pool utility and full backend."""
import argparse
import ast
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[4]
SRT=ROOT/'python/sglang/srt'


def run(output):
    state=types.SimpleNamespace(events=[],active=None,current=None,allocator='native',next_id=0,pools={},saver_calls=0)
    resources=types.SimpleNamespace(graph_memory_pool=None,graph_memory_pool_owner=None,graph_memory_pool_owner_device=None)
    device=types.SimpleNamespace(type='cuda',index=0)
    class Pool:
        def __init__(self):
            state.next_id+=1;self.id=(0,state.next_id);self.count=1;state.pools[self.id]=self
        def use_count(self):return self.count
    class Graph:
        def __init__(self):self.pid=None
        def reset(self):
            if self.pid in state.pools:state.pools[self.pid].count-=1
            self.pid=None
        def replay(self):state.events.append(('replay',self.pid))
        def __del__(self):self.reset()
    @contextmanager
    def stream(s):
        old=state.current;state.current=s
        try:yield
        finally:state.current=old
    @contextmanager
    def use(pool,device=None):
        old=state.active;state.active=pool.id;state.events.append(('pool_enter',pool.id))
        try:yield
        finally:state.events.append(('pool_exit',pool.id));state.active=old
    @contextmanager
    def cuda_graph(cuda_graph,pool=None,stream=None,**kw):
        cuda_graph.pid=pool
        if pool in state.pools:state.pools[pool].count+=1
        state.events.append(('capture_begin',pool))
        yield
        state.events.append(('capture_end',pool))
    @contextmanager
    def cuda_device(_):yield
    cuda=types.SimpleNamespace(MemPool=Pool,use_mem_pool=use,CUDAGraph=Graph,stream=stream,graph=cuda_graph,
        graph_pool_handle=lambda:('legacy',state.next_id),memory=types.SimpleNamespace(get_allocator_backend=lambda:state.allocator),
        current_device=lambda:0,device=cuda_device,synchronize=lambda:state.events.append(('synchronize',)))
    torch=types.ModuleType('torch');torch.cuda=cuda;sys.modules['torch']=torch
    def module(name,**attrs):
        m=types.ModuleType(name);m.__dict__.update(attrs);sys.modules[name]=m;return m
    module('sglang.srt.runtime_context',get_resources=lambda:resources)
    module('sglang.srt.constants',GPU_MEMORY_TYPE_CUDA_GRAPH='graph_tag')
    module('sglang.srt.distributed.device_communicators.pynccl_allocator',set_graph_pool_id=lambda p:state.events.append(('publish_pool',p)))
    module('sglang.srt.model_executor.runner_backend.base_cuda_graph_backend',BaseCudaGraphBackend=object)
    module('sglang.srt.utils',get_bool_env_var=lambda _:True)
    class Saver:
        def __init__(self,enable):self.enabled=enable
        @contextmanager
        def cuda_graph(self,**kwargs):
            state.saver_calls+=1;assert kwargs.pop('tag')=='graph_tag'
            with cuda_graph(**kwargs):yield
    module('sglang.srt.utils.torch_memory_saver_adapter',TorchMemorySaverAdapter=types.SimpleNamespace(create=lambda enable:Saver(enable)))
    def load(path,name):
        sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
    pool=load(SRT/'model_executor/runner_utils/pool.py','sglang.srt.model_executor.runner_utils.pool')
    backend=load(SRT/'model_executor/runner_backend/full_cuda_graph_backend.py','tested_full_backend')
    capture_stream=types.SimpleNamespace(device=device,name='capture')
    runner=types.SimpleNamespace(device_module=cuda,model_runner=types.SimpleNamespace(tp_group=types.SimpleNamespace(barrier=lambda:state.events.append(('barrier',)))),enable_profile_cuda_graph=True,_profiler=types.SimpleNamespace(step=lambda:state.events.append(('profile_step',))))
    shared_output=object()
    def exercise(b,key,expected_route,*,wrong_thread=False):
        state.events.clear();calls=[];hooks=[]
        def forward():
            calls.append((state.active,state.current));return shared_output
        def hook():hooks.append((state.active,state.current))
        with stream(capture_stream),b.capture_session(capture_stream):
            if wrong_thread:b._capture_thread=-1
            b.capture_one(key,forward,post_warmup_hook=hook)
        assert len(calls)==3 and len(hooks)==2
        assert sum(x[0]=='synchronize'for x in state.events)==2
        assert sum(x[0]=='barrier'for x in state.events)==2
        assert sum(x[0]=='profile_step'for x in state.events)==3
        assert (calls[0][0]is not None)==expected_route and (calls[1][0]is not None)==expected_route
        assert all(s is capture_stream for _,s in calls[:2])
        assert all(p is None and s is capture_stream for p,s in hooks)
        assert b._outputs[key]is shared_output
        assert b._capture_stream is None and b._capture_thread is None
        return list(state.events)
    cases=[]
    # An earlier CUDA backend can create the handle: Full adopts its owner.
    pid=pool.get_or_create_global_graph_memory_pool(cuda);owner=resources.graph_memory_pool_owner
    b1=backend.FullCudaGraphBackend(runner);b2=backend.FullCudaGraphBackend(runner)
    exercise(b1,'large',True);exercise(b2,'small',True)
    assert b1._pool==b2._pool==pid and owner.use_count()==3
    assert b1._graphs['large']._sglang_pool_owner is owner
    b1.cleanup();gc.collect();assert resources.graph_memory_pool_owner is owner and owner.use_count()==2
    b2.cleanup();gc.collect();assert owner.use_count()==1 and resources.graph_memory_pool_owner is None
    cases.append('shared_owner_and_last_graph_cleanup')
    # Existing foreign pool IDs cannot be adopted through unsupported private APIs.
    pool.set_global_graph_memory_pool(('external',99));b=backend.FullCudaGraphBackend(runner)
    exercise(b,'legacy',False);b.cleanup();assert pool.get_global_graph_memory_pool()==('external',99)
    cases.append('preexisting_opaque_pool_keeps_original_warmups')
    pool.set_global_graph_memory_pool(None)
    pool.get_or_create_global_graph_memory_pool(cuda)
    resources.graph_memory_pool=('direct_override',1)
    pool.release_unused_global_graph_memory_pool(('direct_override',1))
    assert resources.graph_memory_pool==('direct_override',1)
    pool.set_global_graph_memory_pool(None)
    cases.append('stale_owner_does_not_clear_external_handle')
    state.allocator='cudaMallocAsync';before=state.next_id
    b=backend.FullCudaGraphBackend(runner);exercise(b,'async',False);b.cleanup()
    assert state.next_id==before;state.allocator='native';pool.set_global_graph_memory_pool(None)
    cases.append('non_native_allocator_fallback')
    xpu=types.SimpleNamespace(graph_pool_handle=lambda:('xpu',1));assert pool.get_or_create_global_graph_memory_pool(xpu)==('xpu',1)
    assert resources.graph_memory_pool_owner is None;pool.set_global_graph_memory_pool(None)
    cases.append('non_cuda_handle_unchanged')
    # Saver capture retains the adapter/tag and does not reroute its warmups.
    before=state.next_id;b=backend.FullCudaGraphBackend(runner,enable_memory_saver=True)
    exercise(b,'saver',False);assert state.saver_calls==1 and state.next_id==before;b.cleanup();pool.set_global_graph_memory_pool(None)
    cases.append('memory_saver_path_unchanged')
    # A previously created owner must not enable routing for a saver worker,
    # including when its separate graph-saver switch is off.
    pool.get_or_create_global_graph_memory_pool(cuda)
    b=backend.FullCudaGraphBackend(runner,enable_memory_saver=True)
    b._memory_saver_adapter.enabled=False
    exercise(b,'saver_existing_owner',False);b.cleanup()
    cases.append('existing_owner_does_not_override_raw_saver_flag')
    b=backend.FullCudaGraphBackend(runner);exercise(b,'other_thread',False,wrong_thread=True);b.cleanup()
    cases.append('session_thread_mismatch_does_not_reroute')
    # Caller-held graphs retain their explicit owner across global replacement.
    b=backend.FullCudaGraphBackend(runner);exercise(b,'borrowed',True);g=b._graphs['borrowed'];owner=g._sglang_pool_owner;old=b._pool
    pool.set_global_graph_memory_pool(None);assert g._sglang_pool_owner is owner and owner.use_count()==2
    b.cleanup();assert owner.use_count()==2;g.reset();assert owner.use_count()==1;del g
    cases.append('borrowed_graph_owner_survives_global_reset')
    # Exceptions unwind pool/stream contexts without adding hooks or captures.
    b=backend.FullCudaGraphBackend(runner)
    try:
        with stream(capture_stream),b.capture_session(capture_stream):
            b.capture_one('fail',lambda:(_ for _ in ()).throw(RuntimeError('controlled')))
    except RuntimeError:pass
    else:raise AssertionError('Expected failure')
    assert state.active is None and state.current is None and not b._graphs
    b.cleanup();cases.append('exception_context_cleanup')
    files=['runtime_context.py','model_executor/runner_utils/pool.py','model_executor/runner_backend/full_cuda_graph_backend.py','model_executor/runner_backend/breakable_cuda_graph_backend.py','model_executor/runner_backend/tc_piecewise_cuda_graph_backend.py']
    for name in files[-2:]:
        tree=ast.parse((SRT/name).read_text())
        calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='get_or_create_global_graph_memory_pool']
        assert calls and all(any(k.arg=='allow_explicit_pool'and isinstance(k.value,ast.Attribute)and k.value.attr=='_allow_explicit_pool'for k in c.keywords)for c in calls)
    cases.append('breakable_and_TC_creation_have_explicit_saver_gate')
    result=dict(status='CPU_GRAPH_POOL_PRODUCTION_CONTROL_PASS',cases=cases,GPU_calls=0,checks='Actual utility/backend methods with shape/control-only stand-ins. Real GPU allocation/replay qualification is separate.',source_sha256={str((SRT/n).relative_to(ROOT)):hashlib.sha256((SRT/n).read_bytes()).hexdigest()for n in files})
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


def test_graph_warmup_pool(tmp_path):
    subprocess.run([sys.executable,__file__,'--output',str(tmp_path/'result.json')],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
