use crate::{constants::LOG_PRIME, ring::PolyRingElem};

use rand_chacha::ChaCha8Rng;

#[derive(Clone)]
pub struct PolyVec(pub Vec<PolyRingElem>);

pub struct PolyMatrix(pub Vec<Vec<PolyRingElem>>);

/// Sparse matrix with polynomial coefficients: a vector of
/// `(row_index, column_index, coef)`.
#[allow(dead_code)]
pub struct SparsePolyMatrix(pub Vec<(usize, usize, PolyRingElem)>);

#[allow(dead_code)]
impl PolyVec {
    /// Create an empty `PolyVec`.
    pub fn new() -> Self {
        PolyVec(Vec::new())
    }

    /// Compute scalar product with another `PolyVec`.
    pub fn scalar_prod(&self, other: &PolyVec) -> PolyRingElem {
        let mut result = PolyRingElem::zero();

        for (c_self, c_other) in self.0.iter().zip(other.0.iter()) {
            result += c_self * c_other;
        }

        result
    }

    /// Decompose a `PolyVec` in the base `2^d`. Return a list of `PolyVec`.
    pub fn decomp(&self, d: usize) -> Vec<Self> {
        let length = (LOG_PRIME as usize).div_ceil(d);
        let mut result: Vec<PolyVec> = vec![PolyVec::new(); length];

        for poly in &self.0 {
            for (i, small_poly) in poly.decompose(d).into_iter().enumerate() {
                result[i].0.push(small_poly)
            }
        }

        result
    }

    /// See a `PolyVec` as an array of `u8`, return an iterator over such an array.
    pub fn iter_bytes(&self) -> impl Iterator<Item = u8> {
        self.0
            .iter()
            .map(|poly| {
                poly.element
                    .iter()
                    .map(|base_elem| base_elem.to_le_bytes())
                    .flatten()
            })
            .flatten()
    }

    pub fn zero(dim: usize) -> Self {
        Self(vec![PolyRingElem::zero(); dim])
    }

    pub fn random(dim: usize, rng: &mut ChaCha8Rng) -> Self {
        let mut vec: Vec<PolyRingElem> = Vec::new();

        for _ in 0..dim {
            vec.push(PolyRingElem::random(rng))
        }

        Self(vec)
    }

    pub fn mut_invert_x(&mut self) {
        self.0.iter_mut().for_each(|poly| poly.mut_invert_x());
    }

    pub fn add_assign(&mut self, other: &Self) {
        for (coef_self, coef_other) in self.0.iter_mut().zip(other.0.iter()) {
            *coef_self += coef_other;
        }
    }

    /// Update `self`: `self <- self + coef * other`.
    pub fn add_mul<T: Into<PolyRingElem>>(&mut self, coef: T, other: &Self) {
        let coef_poly: PolyRingElem = coef.into();

        for (coef_self, coef_other) in self.0.iter_mut().zip(other.0.iter()) {
            *coef_self += &coef_poly * coef_other;
        }
    }
}

#[allow(dead_code)]
impl PolyMatrix {
    /// Generate a random matrix of the given size, using the given RNG.
    pub fn random(rng: &mut ChaCha8Rng, rows: usize, cols: usize) -> Self {
        let mut result = vec![vec![PolyRingElem::zero(); cols]; rows];

        result
            .iter_mut()
            .for_each(|row| row.fill_with(|| PolyRingElem::random(rng)));

        Self(result)
    }

    /// Update `output <- output + self * input`.
    pub fn add_apply_raw(&self, output: &mut [PolyRingElem], input: &[PolyRingElem]) {
        if input.len() != self.0[0].len() {
            panic!(
                "Applying matrix to incompatible vector: matrix rows = {}, vector length = {}",
                self.0[0].len(),
                input.len()
            );
        }
        if output.len() != self.0.len() {
            panic!(
                "Storing matrix to incompatible vector: matrix colums = {}, vector length = {}",
                self.0.len(),
                output.len()
            );
        }

        for (i, line) in self.0.iter().enumerate() {
            for (coef, coord) in line.iter().zip(input.iter()) {
                output[i] += coef * coord;
            }
        }
    }

    pub fn apply_raw(&self, input: &[PolyRingElem]) -> Vec<PolyRingElem> {
        let mut output = vec![PolyRingElem::zero(); self.0.len()];
        self.add_apply_raw(&mut output, input);

        output
    }

    /// Update `output <- output + self * input`.
    pub fn add_apply(&self, output: &mut PolyVec, input: &PolyVec) {
        self.add_apply_raw(&mut output.0, &input.0);
    }

    /// Compute `self * input`.
    pub fn apply(&self, input: &PolyVec) -> PolyVec {
        PolyVec(self.apply_raw(&input.0))
    }
}

impl SparsePolyMatrix {
    pub fn new() -> Self {
        Self(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn canonical_basis() {
        // generate random matrix
        let mut seed: <ChaCha8Rng as SeedableRng>::Seed = Default::default();
        rand::rng().fill(&mut seed);
        let mut rng = ChaCha8Rng::from_seed(seed);

        let matrix = PolyMatrix::random(&mut rng, 10, 10);

        // apply matrix to each vector in the canonical basis
        for j in 0..10 {
            let mut vec = vec![PolyRingElem::zero(); 10];
            vec[j] = PolyRingElem::one();

            let result = matrix.apply_raw(&vec);

            for i in 0..10 {
                assert_eq!(result[i], matrix.0[i][j]);
            }
        }
    }
}
