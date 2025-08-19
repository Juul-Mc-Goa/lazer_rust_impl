use std::collections::HashMap;

use crate::{
    constants::{LOG_PRIME, PRIME_BYTES_LEN},
    jl_matrix::JLMatrix,
    linear_algebra::{PolyVec, SparsePolyMatrix},
    ring::{BaseRingElem, PolyRingElem},
};

/// A degree 2 polynomial equation in the `r` witness vectors s_1, ..., s_r in
/// R_p^n:
///
/// ```
/// sum(a_{i,j} <s_i, s_j>) + sum(<b_i, s_i>) + c = 0
/// ```
/// Here:
/// * a_{i,j} is in R_p
/// * b_i is in R_p^n
/// * c is in R_p
///
/// This is stored as one quadratic equation on a single vector `s'`:
/// ```
/// < s', As' > + < B, s' > + c = 0
/// ```
/// where `s' = s_1 || ... || s_r`.
#[allow(dead_code)]
#[derive(Debug)]
pub struct Constraint {
    pub degree: usize,
    pub quadratic_part: SparsePolyMatrix,
    pub linear_part: PolyVec,
    pub constant: PolyRingElem,
}

impl Constraint {
    pub fn new_raw(degree: usize) -> Self {
        Self {
            degree,
            quadratic_part: SparsePolyMatrix(Vec::new()),
            linear_part: PolyVec::new(),
            constant: PolyRingElem::zero(),
        }
    }

    /// Generate a new constraint.
    pub fn new() -> Self {
        Self::new_raw(1)
    }

    /// Join two `Constraint`s into one that works on concatenated vectors.
    ///
    /// If `self.check(v1) == true` and `other.check(v2) == true`, then
    /// `self.join(other).check(v1 || v2) == true`.
    #[allow(dead_code)]
    pub fn join(self, other: Self) -> Self {
        let offset: usize = self.linear_part.0.len();

        let constant = self.constant + other.constant;
        let linear_part = PolyVec::join(&[self.linear_part, other.linear_part]);
        let mut quadratic_part = self.quadratic_part;

        other
            .quadratic_part
            .0
            .into_iter()
            .for_each(|(i, j, coef)| quadratic_part.0.push((i + offset, j + offset, coef)));
        Self {
            degree: self.degree,
            quadratic_part,
            linear_part,
            constant,
        }
    }

    /// Consume `self` to create a new equivalent constraint on the decomposed
    /// witness.
    #[allow(dead_code)]
    pub fn decomp(self, base: usize, len: usize) -> Self {
        let dim = self.linear_part.0.len();

        // quadratic part
        let mut quad_hashmap: HashMap<(usize, usize), PolyRingElem> = HashMap::new();

        for (i, j, coef) in self.quadratic_part.0 {
            // decompose s_i = sum(k, (2 ^ (base * k)) s_ik)
            for k in 0..len {
                let new_i = i + k * dim;

                // decompose s_j = sum(l, (2 ^ (base * l)) s_jl)
                for l in 0..len {
                    let new_j = j + l * dim;

                    let mut to_add = &BaseRingElem::from(1 << (base * (k + l))) * &coef;
                    if k != l {
                        to_add *= BaseRingElem::from(2);
                    }

                    quad_hashmap
                        .entry((new_i, new_j))
                        .and_modify(|c| *c += &to_add)
                        .or_insert(to_add);
                }
            }
        }

        let quadratic_part = SparsePolyMatrix(
            quad_hashmap
                .into_iter()
                .map(|(k, v)| (k.0, k.1, v))
                .collect(),
        );

        // linear part
        let mut to_append = self.linear_part.clone();
        let mut linear_part: Vec<PolyVec> = Vec::with_capacity(len);

        for _ in 0..len {
            linear_part.push(to_append.clone());

            // multiply to_append by 2^base
            to_append *= &BaseRingElem::from(1 << base);
        }

        let linear_part = PolyVec::join(&linear_part);

        Self {
            degree: self.degree,
            quadratic_part,
            linear_part,
            constant: self.constant,
        }
    }

    /// Create 256 linear constraints from a `JLMatrix` array of size `r` and `r` vectors.
    pub fn from_jl_proj(jl_matrices: &[JLMatrix], witness: &[PolyVec]) -> Vec<Self> {
        let mut all_linear_parts: Vec<PolyVec> = vec![PolyVec::new(); 256];
        let mut all_constants: Vec<PolyRingElem> = vec![PolyRingElem::zero(); 256];

        for (i, jl_matrix) in jl_matrices.iter().enumerate() {
            let mut rows = jl_matrix.as_polyvecs_inverted();

            for j in 0..256 {
                all_constants[j] -= rows[j].scalar_prod(&witness[i]);
                all_linear_parts[j].concat(&mut rows[j]);
            }
        }

        all_linear_parts
            .into_iter()
            .zip(all_constants.into_iter())
            .map(|(linear_part, constant)| Self {
                degree: 1,
                quadratic_part: SparsePolyMatrix::new(),
                linear_part,
                constant,
            })
            .collect()
    }
}

/// Aggregate several constraints acting on the same vector space.
pub fn aggregate_constraints(
    r: usize,
    dim: usize,
    constraints: &[Constraint],
    challenges: &[BaseRingElem],
) -> Constraint {
    let mut linear_part: PolyVec = PolyVec::zero(r * dim);
    let mut constant: PolyRingElem = PolyRingElem::zero();

    for (challenge, constraint) in challenges.iter().zip(constraints.iter()) {
        let poly_challenge: PolyRingElem = (*challenge).into();
        // update linear_part
        linear_part.add_mul_assign(&poly_challenge, &constraint.linear_part);
        // update constant
        constant += challenge * &constraint.constant;
    }

    Constraint {
        degree: 1,
        quadratic_part: SparsePolyMatrix::new(),
        linear_part,
        constant,
    }
}

pub fn unpack_challenges(challenges: &[u8]) -> [BaseRingElem; 256] {
    let mut alpha = [BaseRingElem::zero(); 256];

    let build_base_elem = |limbs: &[u8]| -> BaseRingElem {
        let mut shift = 1 << 8_u64;
        let mut result = limbs[0] as u64;
        for i in 1..PRIME_BYTES_LEN {
            result += shift * limbs[i] as u64;
            shift *= 256;
        }

        result.into()
    };

    // 256 = 32 * 8
    let primes_len = LOG_PRIME as usize;
    for i in 0..32 {
        for j in 0..8 {
            alpha[8 * i + j] =
                build_base_elem(&challenges[(primes_len * i + PRIME_BYTES_LEN * j)..]);
        }
    }

    alpha
}
