"""Record the compiler, sources and binaries used by binary-field experiments."""
import hashlib
import json
import pathlib
import subprocess
from datetime import datetime, timezone

def digest(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
nvcc=subprocess.check_output(['nvcc','--version'],text=True).strip()
gcc=subprocess.check_output(['g++','-dumpfullversion','-dumpversion'],text=True).strip()
builds={}
for name,sources,binary,compiler,flags in (
    ('binary_fields',['cuda/binary_fields.cu'],'build/binary.cubin',nvcc,['-O3','-std=c++17','-arch=sm_120','-cubin']),
    ('binary_fused',['cuda/binary_fields.cu','cuda/binary_fused.cu'],'build/binary_fused.cubin',nvcc,['-O3','-std=c++17','-arch=sm_120','-cubin']),
    ('binary_graph',['cuda/binary_fields.cu','cuda/binary_fused.cu','cuda/binary_graph.cu'],'build/binary_graph.cubin',nvcc,['-O3','-std=c++17','-arch=sm_120','-cubin']),
    ('native_tail',['native/binary_tail.cpp'],'build/libbinary_tail.so','g++ '+gcc,['-O3','-std=c++17','-shared','-fPIC','-mpclmul','-msse2'])):
    builds[name]=dict(compiler=compiler,flags=flags,source_sha256={p:digest(p) for p in sources},binary_sha256=digest(binary))
record=dict(utc=datetime.now(timezone.utc).isoformat(),source_sha256={'collect_binary_build.py':digest('collect_binary_build.py')},builds=builds)
pathlib.Path('results/stage2-build.json').write_text(json.dumps(record,indent=2)+'\n')
