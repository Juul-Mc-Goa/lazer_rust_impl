use crate::{
    constants::{LOG_PRIME, PRIME_BYTES_LEN},
    jl_matrix::JLMatrix,
    linear_algebra::{PolyVec, SparsePolyMatrix},
    ring::{BaseRingElem, PolyRingElem},
};

/// A degree 2 polynomial equation in the `r` witness vectors s_1, ..., s_r in
/// R_p^n:
/// ```
/// sum(a_{i,j} <s_i, s_j>) + sum(<b_i, s_i>) + c = 0
/// ```
/// Here:
/// * a_{i,j} is in R_p
/// * b_i is in R_p^n
/// * c is in R_p
#[allow(dead_code)]
pub struct Constraint {
    pub degree: usize,
    pub quadratic_part: SparsePolyMatrix,
    pub linear_part: Vec<PolyVec>,
    pub constant: PolyRingElem,
}

impl Constraint {
    pub fn new_raw(degree: usize) -> Self {
        Self {
            degree,
            quadratic_part: SparsePolyMatrix(Vec::new()),
            linear_part: Vec::new(),
            constant: PolyRingElem::zero(),
        }
    }
    pub fn new() -> Self {
        Self::new_raw(1)
    }

    /// Create 256 linear constraints from a `JLMatrix` array of size `r` and `r` vectors.
    pub fn from_jl_proj(jl_matrices: &[JLMatrix], witness: &[PolyVec], r: usize) -> Vec<Self> {
        let mut all_linear_parts: Vec<Vec<PolyVec>> = vec![vec![PolyVec::new(); r]; 256];
        let mut all_constants: Vec<PolyRingElem> = vec![PolyRingElem::zero(); 256];

        for (i, jl_matrix) in jl_matrices.iter().enumerate() {
            let rows = jl_matrix.as_polyvecs_inverted();

            for j in 0..256 {
                all_constants[j] -= rows[j].scalar_prod(&witness[i]);
                all_linear_parts[j][i].add_assign(&rows[j]);
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

pub fn aggregate_constraints(
    dim: usize,
    r: usize,
    constraints: &[Constraint],
    challenges: &[BaseRingElem],
) -> Constraint {
    let mut linear_part: Vec<PolyVec> = vec![PolyVec::zero(dim); r];
    let mut constant: PolyRingElem = PolyRingElem::zero();

    for (challenge, constraint) in challenges.iter().zip(constraints.iter()) {
        // update linear_part
        for (polyvec_l, polyvec_r) in linear_part.iter_mut().zip(constraint.linear_part.iter()) {
            polyvec_l.add_mul_assign(*challenge, polyvec_r);
        }
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
