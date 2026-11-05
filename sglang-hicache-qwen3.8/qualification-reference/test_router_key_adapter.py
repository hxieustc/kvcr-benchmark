"""Exercise the pinned KVCR strict remote-query contract with the actual adapter."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

class RouterKeyAdapter(unittest.TestCase):
    def test_hint_membership(self):
        root=Path(__file__).resolve().parents[1]/'software'
        tree=ast.parse((root/'sglang/python/sglang/srt/mem_cache/storage/kvcr/host_layout.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StorageKeyAdapter')
        env={};exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'adapter','exec'),env)
        tree=ast.parse((root/'kvcr/src/kvcr/backends/remote_fw_dram.py').read_text())
        method=next(n for cls in tree.body if isinstance(cls,ast.ClassDef) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='query')
        method.returns=None
        for a in method.args.args:a.annotation=None
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'query','exec'),env)
        key=b'8000000000000001'+b'0'*48+b'#namespace#kv'
        hint=NS(source='tcp://peer:1',failed=False,missing_keys=set(),block_hashes={0x8000000000000001})
        remote=NS(_request_hints={'req':hint},_key_adapter=env['StorageKeyAdapter'](),_options=NS(opportunistic_query=False))
        self.assertTrue(env['query'](remote,key,'req'))
        hint.block_hashes.clear();self.assertFalse(env['query'](remote,key,'req'))
        self.assertFalse(env['query'](remote,key,'absent'))

if __name__=='__main__':unittest.main()
