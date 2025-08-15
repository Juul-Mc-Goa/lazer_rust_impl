use crate::constants::{DEGREE, PRIME};
use crate::linear_algebra::PolyVec;
use crate::ring::{BaseRingElem, PolyRingElem};
use crate::utils::Aes128Ctr64LE;

use aes::cipher::StreamCipher;

#[allow(dead_code)]
pub struct JLMatrix {
    pub dim: usize,
    pub data: Vec<u8>,
}

#[allow(dead_code)]
impl JLMatrix {
    pub fn new(dim: usize) -> Self {
        // 1 bit per (in, out) coordinate
        // input: wit.dim[i] * DEGREE coordinates
        // output: 256 coordinates
        let jl_bytes = dim * DEGREE as usize * 256 / 8;

        Self {
            dim,
            data: vec![0_u8; jl_bytes],
        }
    }

    pub fn random(dim: usize, cipher: &mut Aes128Ctr64LE) -> Self {
        // 1 bit per (in, out) coordinate
        // input: wit.dim[i] * DEGREE coordinates
        // output: 256 coordinates
        let jl_bytes = dim * DEGREE as usize * 256 / 8;
        let mut jl_matrix = vec![0_u8; jl_bytes];

        // generate jl_matrix
        cipher.apply_keystream(&mut jl_matrix);

        Self {
            dim,
            data: jl_matrix,
        }
    }

    pub fn add_apply(&self, out_vec: &mut [BaseRingElem; 256], in_vec: &PolyVec) {
        let mut jl_bit_idx = 0;

        // cut data in 256 chunk
        for coord in 0..256 {
            // cut chunk in `dim` sub-chunks
            for v in &in_vec.0 {
                // cut sub-chunk in `DEGREE` sub-sub-chunks
                for coef in &v.element {
                    let byte_idx = jl_bit_idx >> 3;
                    let mask = 1 << (jl_bit_idx & 7);

                    if (self.data[byte_idx] & mask) != 0 {
                        out_vec[coord] -= coef;
                    } else {
                        out_vec[coord] += coef;
                    }

                    jl_bit_idx += 1;
                }
            }
        }
    }

    /// Unpack a `JLMatrix` into a vector of `PolyVec`s (with coefficients equal to +- 1).
    /// Each `PolyVec` represents a line in the matrix.
    pub fn as_polyvecs(&self) -> Vec<PolyVec> {
        let mut result: Vec<PolyVec> = Vec::new();

        for packed_v in self.data.chunks_exact(self.dim) {
            let mut new_v = PolyVec::new();
            for packed_poly in packed_v.chunks_exact(DEGREE as usize >> 3) {
                let mut new_poly_vec: Vec<u64> = Vec::new();

                for bit in 0..(DEGREE as usize) {
                    let byte_idx = bit >> 3;
                    let mask = 1 << (bit & 7);

                    if packed_poly[byte_idx] & mask != 0 {
                        new_poly_vec.push(PRIME - 1);
                    } else {
                        new_poly_vec.push(1);
                    }
                }

                new_v.0.push(PolyRingElem::from_vec_u64(new_poly_vec));
            }

            result.push(new_v);
        }

        result
    }

    /// Apply [`Self::as_polyvecs`], then apply `Polyvec::mut_invert_x` on each
    /// element.
    pub fn as_polyvecs_inverted(&self) -> Vec<PolyVec> {
        let mut result = self.as_polyvecs();

        for v in result.as_mut_slice() {
            v.invert_x();
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::PolyRingElem;

    #[test]
    fn trivial_jl_mat() {
        let mut projection = [BaseRingElem::from(0); 256];
        let in_dim = 4;

        let mut in_vec = PolyVec::zero(in_dim);
        in_vec.0[0] = PolyRingElem::one();
        in_vec.0[1] = PolyRingElem::one();

        let jl_matrix = JLMatrix::new(in_dim);

        jl_matrix.add_apply(&mut projection, &in_vec);

        assert_eq!(projection, [2.into(); 256])
    }

    #[test]
    fn jl_mat_sub_coef() {
        let mut projection = [BaseRingElem::from(0); 256];

        let in_dim = 4;
        let mut in_vec = PolyVec::zero(in_dim);

        let mut first_poly = vec![0_u64; DEGREE as usize];
        first_poly[0] = 1;
        first_poly[1] = 1;
        in_vec.0[0] = PolyRingElem::from_vec_u64(first_poly);

        let mut jl_matrix = JLMatrix::new(in_dim);
        // each 8-byte (64-bit) block is dedicated to one coordinate of
        // projection and one coordinate of in_vec
        jl_matrix.data[0] = 3;

        jl_matrix.add_apply(&mut projection, &in_vec);

        assert_eq!(projection[0], -BaseRingElem::from(2));
        for c in projection[1..].iter() {
            assert_eq!(c, &BaseRingElem::from(2))
        }
    }

    #[test]
    fn jl_mat_sub_vectors() {
        let mut projection = [BaseRingElem::from(0); 256];
        let in_dim = 4;
        let mut in_vec = PolyVec::zero(in_dim);
        in_vec.0[0] = PolyRingElem::one();
        in_vec.0[1] = PolyRingElem::one();

        let mut jl_matrix = JLMatrix::new(in_dim);
        // each 8-byte (= DEGREE bits) block is dedicated to
        // one coordinate in in_vec
        jl_matrix.data[0] = 1;
        jl_matrix.data[8] = 1;

        jl_matrix.add_apply(&mut projection, &in_vec);

        assert_eq!(projection[0], -BaseRingElem::from(2));
        for c in projection[1..].iter() {
            assert_eq!(c, &BaseRingElem::from(2))
        }
    }

    #[test]
    fn jl_mat_out_coord() {
        let mut projection = [BaseRingElem::from(0); 256];

        let in_dim = 4;
        let mut in_vec = PolyVec::zero(in_dim);
        in_vec.0[0] = PolyRingElem::one();
        in_vec.0[1] = PolyRingElem::one();

        let mut jl_matrix = JLMatrix::new(in_dim);
        // each 32-byte (256-bit) block is dedicated to one coordinate of
        // projection and one coordinate of in_vec
        jl_matrix.data[0] = 1;
        jl_matrix.data[32] = 1;

        jl_matrix.add_apply(&mut projection, &in_vec);

        assert_eq!(projection[0], BaseRingElem::from(0));
        assert_eq!(projection[1], BaseRingElem::from(0));
        for c in projection[2..].iter() {
            assert_eq!(c, &BaseRingElem::from(2))
        }
    }
}
