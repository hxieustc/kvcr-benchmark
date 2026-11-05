"""Execute the actual scheduler handshake method without importing GPU runtime."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

class CacheKeyHandshake(unittest.TestCase):
    def test_actual_tree_mode_overrides_speculative_algorithm(self):
        p=Path(__file__).resolve().parents[1]/'software/sglang/python/sglang/srt/managers/scheduler.py'
        tree=ast.parse(p.read_text())
        method=next(n for cls in tree.body if isinstance(cls,ast.ClassDef) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_init_info')
        method.returns=None
        for arg in method.args.args:arg.annotation=None
        mod=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
        env={'get_parallel':lambda:NS(dcp_size=1)};exec(compile(mod,str(p),'exec'),env)
        for mode in (False,True,None):
            worker=NS(max_total_num_tokens=1024,max_req_input_len=512,startup_time=1,page_size=64,
                tree_cache=NS(tree_core=NS(is_eagle=mode),cache_controller=None),
                kv_events_publisher=NS(local_kv_event_sources=lambda *a,**k:[]))
            result=env['get_init_info'](worker)
            if mode is None:self.assertNotIn('cache_key_is_eagle',result)
            else:self.assertIs(result['cache_key_is_eagle'],mode)

if __name__=='__main__':unittest.main()
