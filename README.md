# `lazer_rust_impl`: a LaBRADOR implementation

This project aims at providing an implementation of the [LaBRADOR
protocol](https://eprint.iacr.org/2022/1341). A C implementation has already
been developed and is available
[here](https://github.com/lattice-dogs/labrador/tree/8b6626b26afd4c0162ddd089759d21d3d51bfbdf).

While the C implementation has the advantage of being faster and more complete (with
an additional layer dubbed Lazer described [in this
article](https://eprint.iacr.org/2024/1846) and available
[here](https://github.com/lazer-crypto/lazer)), it requires the `AVX-512`
instruction set extension by Intel.

This project thus aims to implement the protocol for most consumer-grade
computers, at the cost of being less optimized.

## WARNING

This implement is a work in progress, and is not complete yet.
