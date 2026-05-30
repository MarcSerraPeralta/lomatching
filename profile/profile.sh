python -m cProfile -o myLog.profile benchmark_BeliefMoMatching.py
gprof2dot -n 0.1 -e 0.01 -f pstats myLog.profile -o callingGraph.dot
dot -Tsvg callingGraph.dot > output.svg
