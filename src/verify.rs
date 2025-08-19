//! Check that a witness verify the relations stated in a given `Statement`.

use crate::{
    Statement,
    commit::{CommitKeyData, CommitParams},
    linear_algebra::{PolyMatrix, PolyVec, SparsePolyMatrix},
    ring::{BaseRingElem, PolyRingElem},
    utils::{add_apply_matrices_b, add_apply_matrices_garbage, split_tgh},
    witness::Witness,
};

pub enum VerifyError {
    NormCheck {
        norm_square: u128,
        bound_square: u128,
    },
    U1Check(PolyVec),
    U2Check(PolyVec),
    ZCheck(PolyVec),
    PrincipleCheck(PolyRingElem),
    QuadGarbageCheck(PolyRingElem),
    UnifGarbageCheck(PolyRingElem),
}

impl std::fmt::Debug for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use VerifyError::*;
        match self {
            NormCheck {
                norm_square,
                bound_square,
            } => write!(
                f,
                "Norm is bigger than bound: norm = {} (around 2^{}), bound = {} (around 2^{})",
                (*norm_square as f64).sqrt(),
                norm_square.ilog2(),
                (*bound_square as f64).sqrt(),
                bound_square.ilog2(),
            ),
            U1Check(diff) => write!(
                f,
                "u1 != sum(ik, B_ik t_ik) + sum(ijk, C_ijk g_ijk): diff = {diff:?}"
            ),
            U2Check(diff) => write!(f, "u2 != sum(ijk, D_ijk h_ijk): diff = {diff:?}"),
            ZCheck(diff) => write!(f, "Az != sum(i, c_i t_i): diff = {diff:?}"),
            PrincipleCheck(diff) => write!(
                f,
                "sum(ij, a_ij g_ij) + sum(i, h_ii) + c != 0, diff = {diff:?}"
            ),
            QuadGarbageCheck(diff) => write!(f, "<z, z> != sum(ij, c_i c_j g_ij), diff = {diff:?}"),
            UnifGarbageCheck(diff) => {
                write!(
                    f,
                    "<sum(i, c_i b_i), z > != sum(ij, c_i c_j h_ij), diff = {diff:?}"
                )
            }
        }
    }
}

pub type VerifyResult = Result<(), VerifyError>;

/// Unwrap the result of [`Commitments::inner`], assuming it is not `None`.
fn unwrap_tail(garbage_opt: Option<&PolyVec>) -> &PolyVec {
    garbage_opt.expect("statement.tail implies garbage.is_some()")
}

/// Unwrap the result of [`Commitments::outer`], assuming it is not `None`.
fn unwrap_no_tail<'a>(outer_opt: Option<(&'a PolyVec, &'a PolyVec)>) -> (&'a PolyVec, &'a PolyVec) {
    outer_opt.expect("!statement.tail implies outer.is_some()")
}

/// Checks that:
/// * `u1 = matrices_b * t + matrices_c * g`
/// * `u2 = matrices_d * h`
fn check_outer_commits(statement: &Statement, witness: &Witness) -> VerifyResult {
    let CommitKeyData::NoTail {
        matrix_a: _,
        matrices_b,
        matrices_c,
        matrices_d,
    } = &statement.commit_key.data
    else {
        panic!("verify: expected `CommitKeyData::NoTail`");
    };
    let r = statement.r;
    let CommitParams {
        z_base: _,
        z_length: z_len,
        uniform_base: _,
        uniform_length: unif_len,
        quadratic_base: _,
        quadratic_length: quad_len,
        commit_rank_1: _,
        commit_rank_2: _,
        ..
    } = statement.commit_params;
    let com_params = &statement.commit_params;

    let (u1, u2) = unwrap_no_tail(statement.commitments.outer());

    let (t, g, h) = split_tgh(&witness.vectors[z_len], r, com_params);

    // check u1
    let mut diff: PolyVec = u1.clone();
    diff.neg();
    // matrices_b * t
    add_apply_matrices_b(&mut diff.0, matrices_b, t, r, com_params);
    // matrices_c * g
    add_apply_matrices_garbage(&mut diff.0, matrices_c, g, r, quad_len);

    if !diff.is_zero() {
        return Err(VerifyError::U1Check(diff));
    }

    // check u2
    let mut diff: PolyVec = u2.clone();
    diff.neg();
    // matrices_d * h
    add_apply_matrices_garbage(&mut diff.0, matrices_d, h, r, unif_len);

    if !diff.is_zero() {
        return Err(VerifyError::U2Check(diff));
    }

    Ok(())
}

/// Check the following constraint: `Az = sum(i, c_i t_i)`
pub fn check_inner_commit(
    com_rank_1: usize,
    challenges: &[PolyRingElem],
    matrix_a: &PolyMatrix,
    z: &PolyVec,
    t: PolyVec,
) -> VerifyResult {
    let mut diff: PolyVec = matrix_a.apply(&z);
    diff.neg(); // diff = -Az
    // diff += sum(i, c_i t_i)
    challenges
        .iter()
        .zip(t.into_chunks(com_rank_1).iter())
        .for_each(|(c_i, t_i)| diff.add_mul_assign(c_i, t_i));

    if !diff.is_zero() {
        return Err(VerifyError::ZCheck(diff));
    }

    Ok(())
}

/// Check the following constraint: `< z, z > = sum(ij, c_i c_j g_ij)`.
pub fn check_quad_constraint(
    r: usize,
    challenges: &[PolyRingElem],
    z: &PolyVec,
    g: &PolyVec,
) -> VerifyResult {
    let mut diff = -z.scalar_prod(&z);

    (0..r).for_each(|i| {
        let quad_i = i * (i + 1) / 2;
        (0..i).for_each(|j| {
            let mut term = &BaseRingElem::from(2) * &challenges[i];
            term *= &(&challenges[j] * &g.0[quad_i + j]);
            diff += &term;
        });
        let term = &challenges[i] * &challenges[i] * &g.0[quad_i + i];
        diff += &term;
    });

    if !diff.is_zero() {
        return Err(VerifyError::QuadGarbageCheck(diff));
    }

    Ok(())
}

/// Check the following constraint: `< sum(i, c_i b_i), z > = sum(ij, c_i c_j h_ij)`.
pub fn check_h_constraint(
    r: usize,
    dim: usize,
    z: &PolyVec,
    challenges: &[PolyRingElem],
    linear_part: &PolyVec,
    h: &PolyVec,
) -> VerifyResult {
    let linear_part = linear_part.clone().into_chunks(dim);
    let mut diff = PolyRingElem::zero();

    // -sum(i, c_i b_i)
    let mut lin_aggregate = PolyVec::zero(dim);
    linear_part
        .iter()
        .zip(challenges.iter())
        .for_each(|(b_i, c_i)| lin_aggregate.add_mul_assign(c_i, b_i));
    lin_aggregate.neg();

    diff += lin_aggregate.scalar_prod(z);

    // sum(ij, c_i c_j h_ij)
    let mut h_part: PolyVec = PolyVec::zero(r * (r + 1) / 2);
    (0..r).for_each(|i| {
        let quad_i = i * (i + 1) / 2;
        (0..=i).for_each(|j| {
            h_part.0[quad_i + j] = &challenges[i] * &challenges[j];
            diff += &challenges[i] * &challenges[j] * &h.0[quad_i + j];
        })
    });
    if !diff.is_zero() {
        return Err(VerifyError::UnifGarbageCheck(diff));
    }
    Ok(())
}

#[allow(dead_code)]
pub fn check_principle(
    r: usize,
    g: &PolyVec,
    h: &PolyVec,
    quadratic_part: &SparsePolyMatrix,
    constant: &PolyRingElem,
) -> VerifyResult {
    let mut diff: PolyRingElem = constant.clone();

    // diff = sum(ij, a_ij g_ij)
    diff += quadratic_part.apply_to_garbage(g);

    // diff += sum(i, h_ii)
    let mut sum_h_ii = PolyRingElem::zero();
    (0..r).for_each(|i| {
        let quad_idx = i * (i + 1) / 2 + i;

        diff += &h.0[quad_idx];
        sum_h_ii += &h.0[quad_idx];
    });

    if !diff.is_zero() {
        return Err(VerifyError::PrincipleCheck(diff));
    }

    Ok(())
}

pub fn verify(output_stat: &Statement, input_stat: &Statement, witness: &Witness) -> VerifyResult {
    let norm_square: u128 = witness.vectors.iter().map(|w| w.norm_square()).sum();

    if norm_square > output_stat.squared_norm_bound {
        return Err(VerifyError::NormCheck {
            norm_square,
            bound_square: output_stat.squared_norm_bound,
        });
    }

    let r = output_stat.r;
    let z_base = output_stat.commit_params.z_base;
    let z_len = output_stat.commit_params.z_length;
    let unif_base = output_stat.commit_params.uniform_base;
    let unif_len = output_stat.commit_params.uniform_length;
    let quad_base = output_stat.commit_params.quadratic_base;
    let quad_len = output_stat.commit_params.quadratic_length;
    let com_rank_1 = output_stat.commit_params.commit_rank_1;

    let inner: &PolyVec = output_stat.commitments.inner();
    let garbage: Option<&PolyVec> = output_stat.commitments.garbage();
    let challenges: &[PolyRingElem] = &input_stat.challenges;

    // last witness (no tail): t_part || g_part || h_part
    let z: PolyVec;
    let t: PolyVec;
    let g: PolyVec;
    let h: PolyVec;

    if output_stat.tail {
        z = PolyVec::new();
        g = PolyVec::new();
        t = inner.clone();
        h = unwrap_tail(garbage).clone();
    } else {
        let _ = check_outer_commits(output_stat, witness)?;

        let long_z = PolyVec::join(&witness.vectors[..z_len]);
        let tgh = witness.vectors[z_len].clone();

        let quad_size = r * (r + 1) / 2;
        let t_len = unif_len * r * com_rank_1;
        let g_len = quad_len * quad_size;

        let (long_t, gh) = tgh.split(t_len);
        let (long_g, long_h) = gh.split(g_len);

        z = long_z.recompose(z_base as u64, z_len);
        t = long_t.recompose(unif_base as u64, unif_len);
        g = long_g.recompose(quad_base as u64, quad_len);
        h = long_h.recompose(unif_base as u64, unif_len);
    }

    let matrix_a: &PolyMatrix = match &output_stat.commit_key.data {
        CommitKeyData::NoTail { matrix_a, .. } => &matrix_a,
        _ => panic!("verify: commit key data is tail"),
    };

    check_inner_commit(com_rank_1, challenges, &matrix_a, &z, t)?;

    let quadratic_part = &input_stat.constraint.quadratic_part;
    let constant = &input_stat.constraint.constant;

    // < z, z > = sum(ij, c_i c_j g_ij)
    if !quadratic_part.0.is_empty() {
        check_quad_constraint(r, challenges, &z, &g)?;
    }

    // < sum(i, c_i b_i), z > = sum(ij, c_i c_j h_ij)
    check_h_constraint(
        r,
        output_stat.dim,
        &z,
        challenges,
        &input_stat.constraint.linear_part,
        &h,
    )?;

    check_principle(r, &g, &h, quadratic_part, constant)?;

    Ok(())

    // // aggregated relation
    // let new_wit = {
    //     // split z
    //     let mut vectors: Vec<PolyVec> = witness.vectors[..z_len]
    //         .into_iter()
    //         .map(|v| v.clone().into_chunks(output_stat.dim))
    //         .flatten()
    //         .collect();
    //     // split tgh
    //     vectors.append(&mut witness.vectors[z_len].clone().into_chunks(output_stat.dim));

    //     let r = vectors.len();
    //     let dim = vec![output_stat.dim; r];
    //     let norm_square: Vec<u128> = vectors.iter().map(|v| v.norm_square()).collect();

    //     Witness {
    //         r,
    //         dim,
    //         norm_square,
    //         vectors,
    //     }
    // };

    // let result = output_stat.constraint.check(&new_wit);
    // if result.is_zero() {
    //     Ok(())
    // } else {
    //     Err(VerifyError::PrincipleCheck(result))
    // }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_constraint() {
        let r = 3;
        let zero = PolyRingElem::zero();
        let one = PolyRingElem::one();
        let three = &BaseRingElem::from(3) * &one;
        let wit = vec![
            PolyVec(vec![one.clone(); 3]),
            PolyVec(vec![one.clone(), zero.clone(), zero.clone()]),
            PolyVec(vec![zero.clone(), zero.clone(), one.clone()]),
        ];
        let g = PolyVec(vec![
            three.clone(),
            one.clone(),
            one.clone(),
            one.clone(),
            zero.clone(),
            one.clone(),
        ]);

        let challenges = PolyVec(vec![one.clone(); 3]);
        let z = wit[0].clone() + &wit[1] + &wit[2];

        let result = check_quad_constraint(r, &challenges.0, &z, &g);
        println!("{result:?}");

        assert!(result.is_ok());
    }
}
