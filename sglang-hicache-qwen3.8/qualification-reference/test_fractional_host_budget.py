"""CPU CLI/IPC contract for the requested 20% host budget; no model allocation."""
import argparse
import unittest
import msgspec
from sglang.srt.arg_groups.arg_utils import add_cli_args_from_dataclass
from sglang.srt.arg_groups.fields.memory import Memory

HOST_BYTES=(128*2**30)//5

class FractionalHostBudget(unittest.TestCase):
    def test_cli_accepts_fractional_decimal_gigabytes(self):
        parser=argparse.ArgumentParser()
        add_cli_args_from_dataclass(parser,Memory,fields=['hicache_size'])
        args=parser.parse_args(['--hicache-size',str(HOST_BYTES/1e9)])
        self.assertEqual(round(args.hicache_size*1e9),HOST_BYTES)

    def test_wire_roundtrip_preserves_requested_host_budget(self):
        payload=msgspec.json.encode({'hicache_size':HOST_BYTES/1e9})
        memory=msgspec.json.decode(payload,type=Memory)
        self.assertEqual(round(memory.hicache_size*1e9),HOST_BYTES)

if __name__=='__main__':unittest.main()
