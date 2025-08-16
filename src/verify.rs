//! Check that a witness verify the relations stated in a given `Statement`.

use crate::{
    Statement,
    commit::{CommitKeyData, CommitParams},
    constraint::Constraint,
    linear_algebra::{PolyMatrix, PolyVec},
    ring::PolyRingElem,
    witness::Witness,
};

/// Unwrap the result of [`Commitments::inner`], assuming it is not `None`.
fn unwrap_tail(garbage_opt: Option<&PolyVec>) -> &PolyVec {
    garbage_opt.expect("statement.tail implies garbage.is_some()")
}

/// Unwrap the result of [`Commitments::outer`], assuming it is not `None`.
fn unwrap_no_tail<'a>(outer_opt: Option<(&'a PolyVec, &'a PolyVec)>) -> (&'a PolyVec, &'a PolyVec) {
    outer_opt.expect("!statement.tail implies outer.is_some()")
}

/// Checks that:
/// * `u1 = matrices_b * inner + matrices_c * quad_garbage`
/// * `u2 = matrices_d * unif_garbage`
fn check_commits(statement: &Statement, witness: &Witness) {
    let CommitKeyData::NoTail {
        matrix_a: _,
        matrices_b,
        matrices_c: _,
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
        commit_rank_1: com_rank_1,
        commit_rank_2: _,
        ..
    } = statement.commit_params;

    let inner: &[PolyRingElem] = &statement.commitments.inner().0;
    let (u1, u2) = unwrap_no_tail(statement.commitments.outer());
    let inner_len = unif_len * r * com_rank_1;
    let garbage_len = quad_len * r * (r + 1) / 2;
    let h_start = inner_len + garbage_len;
    let h: &[PolyRingElem] = &witness.vectors[z_len].0[h_start..];

    // check u1
    let mut diff: PolyVec = u1.clone();
    diff.neg();
    for k in 0..unif_len {
        for i in 0..r {
            let start = (k * r + i) * com_rank_1;
            let end = start + com_rank_1;
            matrices_b[k][i].add_apply_raw(&mut diff.0, &inner[start..end]);
        }
    }

    assert!(diff.is_zero(), "verify: u1 != matrices_b * inner");

    // check u2
    let mut diff: PolyVec = u2.clone();
    diff.neg();
    for k in 0..unif_len {
        for i in 0..r {
            let start = i * (i + 1) / 2;
            let end = start + i;
            matrices_d[k][i].add_apply_raw(&mut diff.0, &h[start..end]);
        }
    }

    assert!(diff.is_zero(), "verify: u2 != matrices_d * unif_garbage");
}

pub fn verify(statement: &Statement, witness: &Witness) -> bool {
    let norm_square: u128 = witness.vectors.iter().map(|w| w.norm_square()).sum();
    assert!(
        norm_square < statement.squared_norm_bound,
        "verify: witness norm bigger than bound"
    );

    let r = statement.r;
    let z_base = statement.commit_params.z_base;
    let z_len = statement.commit_params.z_length;
    let unif_base = statement.commit_params.uniform_base;
    let unif_len = statement.commit_params.uniform_length;
    let quad_base = statement.commit_params.quadratic_base;
    let quad_len = statement.commit_params.quadratic_length;
    let com_rank_1 = statement.commit_params.commit_rank_1;

    let mut inner: &PolyVec = statement.commitments.inner();
    let mut garbage: Option<&PolyVec> = statement.commitments.garbage();
    let challenges: &[PolyRingElem] = &statement.challenges;
    // let mut outer: Option<(&PolyVec, &PolyVec)> = statement.commitments.outer();

    // last witness:
    //   no tail: z_part || t_part || g_part || h_part
    //   tail:    t_part || g_part || h_part
    let mut z_part: PolyVec;
    let mut t_part: PolyVec;
    let mut g_part: PolyVec;
    let mut h_part: PolyVec;

    if statement.tail {
        z_part = PolyVec::new();
        t_part = inner.clone();
        g_part = PolyVec::new();
        h_part = unwrap_tail(garbage).clone();
    } else {
        check_commits(statement, witness);
        let quad_size = r * (r + 1) / 2;

        let mut long_z = PolyVec::new();
        witness.vectors[..z_len]
            .iter()
            .for_each(|w| long_z.0.extend_from_slice(&w.0));

        let mut long_t = PolyVec::new();
        let t_end = z_len + unif_len * r * com_rank_1;
        witness.vectors[z_len..t_end]
            .iter()
            .for_each(|w| long_t.0.extend_from_slice(&w.0));

        let mut long_g = PolyVec::new();
        let g_end = t_end + quad_len * quad_size;
        witness.vectors[t_end..g_end]
            .iter()
            .for_each(|w| long_g.0.extend_from_slice(&w.0));

        let mut long_h = PolyVec::new();
        witness.vectors[g_end..]
            .iter()
            .for_each(|w| long_h.0.extend_from_slice(&w.0));

        z_part = long_z.recompose(z_base as u64, z_len);
        t_part = long_t.recompose(unif_base as u64, unif_len);
        g_part = long_g.recompose(quad_base as u64, quad_len);
        h_part = long_h.recompose(unif_base as u64, unif_len);
    }

    // Az = sum(i, c_i t_i)
    use CommitKeyData::*;
    let matrix_a: &PolyMatrix = match &statement.commit_key.data {
        Tail { matrix_a, .. } => matrix_a,
        NoTail { matrix_a, .. } => matrix_a,
    };

    let mut diff: PolyVec = matrix_a.apply(&z_part);
    diff.neg(); // diff = -Az
    // diff += sum(i, c_i t_i)
    challenges
        .iter()
        .zip(t_part.into_chunks(com_rank_1).iter())
        .for_each(|(c_i, t_i)| diff.add_mul_assign(c_i, t_i));

    assert!(diff.is_zero(), "verify: Az != sum(i, c_i t_i)");

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

    assert!(
        diff.is_zero(),
        "verify: sum(ij, a_ij g_ij) + sum(i, h_ii) + c != 0"
    );

    // < z, z > = sum(ij, c_i c_j g_ij)
    if !statement.constraint.quadratic_part.0.is_empty() {
        let mut diff = -z_part.scalar_prod(&z_part);
        (0..r).for_each(|i| {
            (0..=i).for_each(|j| {
                diff += &challenges[i] * &challenges[j] * &g_part.0[i * (i + 1) / 2 + j]
            })
        });

        assert!(diff.is_zero(), "verify: < z, z > != sum(ij, c_i c_j g_ij)");
    }

    // < sum(i, c_i b_i), z > = sum(ij, c_i c_j h_ij)
    let mut lin_aggregate = PolyVec::zero(statement.dim);
    (0..r).for_each(|i| lin_aggregate += &linear_part[i] * &challenges[i]);
    lin_aggregate.neg();

    (0..r).for_each(|i| {
        (0..=i)
            .for_each(|j| diff += &challenges[i] * &challenges[j] * &h_part.0[i * (i + 1) / 2 + j])
    });

    assert!(
        diff.is_zero(),
        "verify: < sum(i, c_i b_i), z > != sum(ij, c_i c_j h_ij)"
    );
    todo!()
}
