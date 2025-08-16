use std::ops::{Add, AddAssign, Mul, MulAssign, Neg};

use crate::ring::{BaseRingElem, PolyRingElem};

use rand_chacha::ChaCha8Rng;

#[derive(Clone, Debug)]
pub struct PolyVec(pub Vec<PolyRingElem>);

pub struct PolyMatrix(pub Vec<Vec<PolyRingElem>>);

/// Sparse matrix with polynomial coefficients: a vector of
/// `(row_index, column_index, coef)`.
#[allow(dead_code)]
#[derive(Debug)]
pub struct SparsePolyMatrix(pub Vec<(usize, usize, PolyRingElem)>);

#[allow(dead_code)]
impl PolyVec {
    /// Create an empty `PolyVec`.
    pub fn new() -> Self {
        PolyVec(Vec::new())
    }

    /// Create a null `PolyVec`.
    pub fn zero(dim: usize) -> Self {
        Self(vec![PolyRingElem::zero(); dim])
    }

    /// Compute the sum of each squared coefficients (in `A_q`)
    pub fn norm_square(&self) -> u128 {
        self.0.iter().map(|r| r.norm_square()).sum()
    }

    /// Check if this `PolyVec` is the zero vector.
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|coord| coord.is_zero())
    }

    /// Compute scalar product with another `PolyVec`.
    pub fn scalar_prod(&self, other: &PolyVec) -> PolyRingElem {
        let mut result = PolyRingElem::zero();

        for (c_self, c_other) in self.0.iter().zip(other.0.iter()) {
            result += c_self * c_other;
        }

        result
    }

    /// Decompose a `PolyVec` in the base `2^base`. Return a list of `len` `PolyVec`s.
    pub fn decomp(&self, base: usize, len: usize) -> Vec<Self> {
        let mut result: Vec<PolyVec> = vec![PolyVec::new(); len];

        for poly in &self.0 {
            for (i, small_poly) in poly.decompose(base, len).into_iter().enumerate() {
                result[i].0.push(small_poly)
            }
        }

        result
    }

    /// Recompose a short `PolyVec` from a long one containing its decomposition.
    pub fn recompose(self, base: u64, length: usize) -> PolyVec {
        assert!(
            self.0.len() % length == 0,
            "recompose PolyVec: {} is not a multiple of {length}",
            self.0.len()
        );

        let mut base_power: BaseRingElem = base.into();
        let mut result = PolyVec::zero(self.0.len() / length);
        for small_self in self.into_chunks(length).into_iter() {
            result += &(small_self * &base_power);
            base_power = &base_power * BaseRingElem::from(base);
        }

        result
    }

    /// Concatenate two `PolyVec`s: `self <- self || other`.
    pub fn concat(&mut self, other: &mut PolyVec) {
        self.0.append(&mut other.0);
    }

    pub fn split(self, idx: usize) -> (Self, Self) {
        let tmp = self.0.split_at(idx);

        (PolyVec(tmp.0.to_vec()), PolyVec(tmp.1.to_vec()))
    }

    /// Split one `PolyVec` into chunks of size `split_dim`. The last chunk is padded with
    /// zeros.
    pub fn into_chunks(self, split_dim: usize) -> Vec<PolyVec> {
        let mut result: Vec<PolyVec> = Vec::new();
        let regular_chunks = self.0.len() / split_dim;

        for i in 0..regular_chunks {
            let (start, end) = (split_dim * i, split_dim * (i + 1));
            result.push(PolyVec(self.0[start..end].to_vec()));
        }

        if self.0.len() % split_dim != 0 {
            let start = split_dim * regular_chunks;
            let mut last_chunk = self.0[start..].to_vec();
            last_chunk.resize(split_dim, PolyRingElem::zero());
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

    /// Generate an uniformly random `PolyVec` from a given RNG.
    pub fn random(dim: usize, rng: &mut ChaCha8Rng) -> Self {
        let mut vec: Vec<PolyRingElem> = Vec::new();

        for _ in 0..dim {
            vec.push(PolyRingElem::random(rng))
        }

        Self(vec)
    }

    pub fn invert_x(&mut self) {
        self.0.iter_mut().for_each(|poly| poly.invert_x());
    }

    pub fn neg(&mut self) {
        self.0.iter_mut().for_each(|poly| poly.neg());
    }

    /// Update `self`: `self <- self + coef * other`.
    pub fn add_mul_assign(&mut self, coef: &PolyRingElem, other: &Self) {
        for (coef_self, coef_other) in self.0.iter_mut().zip(other.0.iter()) {
            *coef_self += coef * coef_other;
        }
    }
}

impl<'a> AddAssign<&'a PolyVec> for PolyVec {
    fn add_assign(&mut self, other: &'a PolyVec) {
        for (self_coord, other_coord) in self.0.iter_mut().zip(other.0.iter()) {
            *self_coord += other_coord;
        }
    }
}
impl AddAssign<PolyVec> for PolyVec {
    fn add_assign(&mut self, other: PolyVec) {
        *self += &other;
    }
}

impl<T> Add<T> for PolyVec
where
    PolyVec: AddAssign<T>,
{
    type Output = PolyVec;
    fn add(self, other: T) -> PolyVec {
        let mut new = self.clone();
        new += other;

        new
    }
}

/// Scalar multiplication.
impl<'a, T> MulAssign<&'a T> for PolyVec
where
    PolyRingElem: MulAssign<&'a T>,
{
    fn mul_assign(&mut self, other: &'a T) {
        for coef_self in self.0.iter_mut() {
            *coef_self *= other;
        }
    }
}

impl<'a, T> Mul<&'a T> for &PolyVec
where
    PolyVec: MulAssign<&'a T>,
{
    type Output = PolyVec;
    fn mul(self, other: &'a T) -> Self::Output {
        let mut new = self.clone();
        new *= other;

        new
    }
}

impl<'a, T> Mul<&'a T> for PolyVec
where
    for<'b> &'b PolyVec: Mul<&'a T, Output = PolyVec>,
{
    type Output = PolyVec;
    fn mul(self, other: &'a T) -> PolyVec {
        &self * other
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
                "Storing matrix mul result to incompatible vector: matrix colums = {}, vector length = {}",
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

    /// Compute `self * input`.
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

    /// Compute `transpose(self) * input`.
    pub fn apply_transpose_raw(&self, input: &[PolyRingElem]) -> Vec<PolyRingElem> {
        let (height, width) = (self.0.len(), self.0[0].len());
        let mut result = vec![PolyRingElem::zero(); width];

        if input.len() != height {
            panic!(
                "Apply transpose: matrix of size ({height}, {width}), vector of size {}",
                input.len()
            );
        }

        for (coef, line) in input.iter().zip(self.0.iter()) {
            for (result_coord, input_coord) in result.iter_mut().zip(line.iter()) {
                *result_coord += coef * input_coord;
            }
        }

        result
    }

    /// Compute `transpose(self) * input`.
    pub fn apply_transpose(&self, input: &PolyVec) -> PolyVec {
        PolyVec(self.apply_transpose_raw(&input.0))
    }
}

impl SparsePolyMatrix {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn apply(&self, packed_coefs: &PolyVec, r: usize) -> PolyRingElem {
        let mut result = PolyRingElem::zero();
        self.0
            .iter()
            .for_each(|(i, j, coef)| result += coef * &packed_coefs.0[i * (i + 1) / 2 + j]);

        result
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
