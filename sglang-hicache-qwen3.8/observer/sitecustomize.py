import os
if os.environ.get('HICACHE_BENCH_OUTPUT'):
    from hicache_observer import install
    install()
