# To Do

- [ ] Check if altering adapt code is feasible within timeframe

## Structs for GPU (only if altering adapt code is not possible)
- [ ] Determine general method to alter structs for GPU
- [ ] Write notes on how meshing structs need to be altered for GPU and discuss  practical possibilities with Oscar
- [ ] Alter field structs for GPU
- [ ] Alter model framework structs for GPU
- [ ] Alter discretise structs for GPU
- [ ] Alter preconditioner structs for GPU
- [ ] Alter RANS model structs for GPU
- [ ] Alter simulate structs for GPU
- [ ] Alter UNV structs for GPU
- [ ] Test structs against simple addition kernel

Note - struct alteration will only happen if structs cannot be read within simple addition test kernel

### Kernels
- [ ] Draw flowchart for generic kernel using FVM
- [ ] Write basic flux kernel and note any points of understanding for writing kernels in this program
- [ ] Write the rest of the basic kernels
- [ ] Optimise kernels for shared memory and concurrent execution where possible
