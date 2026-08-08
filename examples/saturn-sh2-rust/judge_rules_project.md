7. ADDRESS IN FUNCTION NAME: a `fn` whose name embeds an 8-hex-digit
address (e.g. `fn_0600beec_something`). Naming must be purely descriptive;
the address belongs only in a doc comment or the dispatch-table entry,
never the identifier.
8. `Sh2Cpu`/`SaturnMemory`-SPECIFIC RAW POINTER: any Saturn address treated
as a real host pointer (`some_addr as *mut u8` + `.write_volatile()`/
`.read_volatile()`). Must always go through `MemoryMap`'s `read_u8`/
`write_u32`/etc, which take a plain `u32` address, never a pointer - this
is the same as the generic raw-pointer rule, restated because it was a real,
previously-hit bug specific to this project's `SaturnMemory` abstraction.
