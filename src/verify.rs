//! Check that a witness verify the relations stated in a given `Statement`.

use crate::{
    Statement,
    commit::{CommitKeyData, CommitParams},
    constraint::Constraint,
    linear_algebra::{PolyMatrix, PolyVec},
    ring::PolyRingElem,
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
                write!(f, "<sum(i, c_i b_i), z > != sum(ij, h_ij), diff = {diff:?}")
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
fn check_commits(statement: &Statement, witness: &Witness) -> VerifyResult {
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

pub fn verify(statement: &Statement, witness: &Witness) -> VerifyResult {
    let norm_square: u128 = witness.vectors.iter().map(|w| w.norm_square()).sum();

    if norm_square < statement.squared_norm_bound {
        return Err(VerifyError::NormCheck {
            norm_square,
            bound_square: statement.squared_norm_bound,
        });
    }

    let r = statement.r;
    let z_base = statement.commit_params.z_base;
    let z_len = statement.commit_params.z_length;
    let unif_base = statement.commit_params.uniform_base;
    let unif_len = statement.commit_params.uniform_length;
    let quad_base = statement.commit_params.quadratic_base;
    let quad_len = statement.commit_params.quadratic_length;
    let com_rank_1 = statement.commit_params.commit_rank_1;

    let inner: &PolyVec = statement.commitments.inner();
    let garbage: Option<&PolyVec> = statement.commitments.garbage();
    let challenges: &[PolyRingElem] = &statement.challenges;

    // last witness (no tail): t_part || g_part || h_part
    let z_part: PolyVec;
    let t_part: PolyVec;
    let g_part: PolyVec;
    let h_part: PolyVec;

    if statement.tail {
        z_part = PolyVec::new();
        t_part = inner.clone();
        g_part = PolyVec::new();
        h_part = unwrap_tail(garbage).clone();
    } else {
        let _ = check_commits(statement, witness)?;

        let long_z = PolyVec::join(&witness.vectors[..z_len]);
        let tgh = witness.vectors[z_len].clone();

        let quad_size = r * (r + 1) / 2;
        let t_len = unif_len * r * com_rank_1;
        let g_len = quad_len * quad_size;

        let (long_t, gh) = tgh.split(t_len);
        let (long_g, long_h) = gh.split(g_len);

        z_part = long_z.recompose(z_base as u64, z_len);
        t_part = long_t.recompose(unif_base as u64, unif_len);
        g_part = long_g.recompose(quad_base as u64, quad_len);
        h_part = long_h.recompose(unif_base as u64, unif_len);
    }

    // Az = sum(i, c_i t_i)
    use CommitKeyData::*;
    let matrix_a: &PolyMatrix = match &statement.commit_key.data {
        NoTail { matrix_a, .. } => matrix_a,
        _ => panic!("verify: commit key data is tail"),
    };

    println!("z: {z_part:?}");
    let mut diff: PolyVec = matrix_a.apply(&z_part);
    diff.neg(); // diff = -Az
    // diff += sum(i, c_i t_i)
    challenges
        .iter()
        .zip(t_part.into_chunks(com_rank_1).iter())
        .for_each(|(c_i, t_i)| diff.add_mul_assign(c_i, t_i));

    if !diff.is_zero() {
        return Err(VerifyError::ZCheck(diff));
    }

    // sum(ij, a_ij g_ij) + sum(i, h_ii) + c = 0
    let Constraint {
        degree: _,
        quadratic_part,
        linear_part,
        constant,
    } = &statement.constraint;
    // diff = c
    let mut diff: PolyRingElem = constant.clone();

    // diff += sum(ij, a_ij g_ij)
    diff += quadratic_part.apply(&g_part, r);

    // diff += sum(i, h_ii)
    (0..r).for_each(|i| diff += &h_part.0[i * (i + 1) / 2 + i]);

    if !diff.is_zero() {
        return Err(VerifyError::PrincipleCheck(diff));
    }

    // < z, z > = sum(ij, c_i c_j g_ij)
    if !statement.constraint.quadratic_part.0.is_empty() {
        let mut diff = -z_part.scalar_prod(&z_part);
        (0..r).for_each(|i| {
            (0..=i).for_each(|j| {
                diff += &challenges[i] * &challenges[j] * &g_part.0[i * (i + 1) / 2 + j]
            })
        });

        if !diff.is_zero() {
            return Err(VerifyError::QuadGarbageCheck(diff));
        }
    }

    // < sum(i, c_i b_i), z > = sum(ij, c_i c_j h_ij)
    let mut diff = PolyRingElem::zero();

    // sum(i, c_i b_i)
    let mut lin_aggregate = PolyVec::zero(statement.dim);
    (0..r).for_each(|i| lin_aggregate += &linear_part[i] * &challenges[i]);
    lin_aggregate.neg();

    diff += lin_aggregate.scalar_prod(&z_part);

    // sum(ij, c_i c_j h_ij)
    (0..r).for_each(|i| {
        (0..=i)
            .for_each(|j| diff += &challenges[i] * &challenges[j] * &h_part.0[i * (i + 1) / 2 + j])
    });

    // "verify: < sum(i, c_i b_i), z > != sum(ij, c_i c_j h_ij)"
    if !diff.is_zero() {
        return Err(VerifyError::UnifGarbageCheck(diff));
    }

    Ok(())
}
