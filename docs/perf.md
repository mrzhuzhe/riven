# Perf 

## ./gemm/src

1. perf record -e cpu-cycles --call-graph fp -o ./outputs/test_MMult.data ./outputs/test_MMult.x
2. perf report -i ./outputs/test_MMult.data
3. rm ./outputs/test_MMult.data

### Notes 
1. perf can show call graph without -g
2. but packA packB is not include in graph
3. dwarf seem better than fp