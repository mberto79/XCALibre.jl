# To Do

- [x] Check if altering adapt code is feasible within timeframe - NOT POSSIBLE

## Structs for GPU (only if altering adapt code is not possible)
- [x] Determine general method to alter structs for GPU - SEE METHOD BELOW
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

#### Struct Alteration Method
1. Write a similar struct to the one being examined using named tuples instead of vectors
2. Set the number of elements in each tuple to be constant based on the type used (e.g., Int8, Int16, etc.)
3. Populate the tuples with two instances of data for each tyep defined in the struct
4. Augment the named tuple structure and struct contents so it can load on an addition or copy-matrix kernel