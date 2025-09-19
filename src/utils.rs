use crate::{
    commit::CommitParams,
    linear_algebra::{PolyMatrix, PolyVec},
    ring::PolyRingElem,
};

use rayon::prelude::*;

pub type Aes128Ctr64LE = ctr::Ctr64LE<aes::Aes128>;

/// Computes smallest power of 2 that is not smaller than `x`.
pub fn next_2_power(mut x: u64) -> u64 {
    x -= 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    x += 1;

    x
}

pub fn print_header(title: &str) {
    let width = 100;
    let pad_size = width - title.len() - 4;
    let half_size = pad_size / 2;

    let left_pad = "═".repeat(half_size);
    let right_pad = "═".repeat(pad_size - half_size);
    println!("\n{left_pad}╡ {title} ╞{right_pad}");
}

pub fn print_sub_header(title: &str) {
    let width = 100;
    let pad_size = width - title.len() - 4;
    let half_size = pad_size / 2;

    let left_pad = "─".repeat(half_size);
    let right_pad = "─".repeat(pad_size - half_size);
    println!("{left_pad}┤ {title} ├{right_pad}");
}

pub fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut result = String::new();
    for byte in bytes {
        result += &format!("{:02x}", byte);
    }

    result
}

/// Split the last witness (`t || g || h`).
pub fn split_tgh<'a>(
    tgh: &'a PolyVec,
    r: usize,
    com_params: &CommitParams,
) -> (&'a [PolyRingElem], &'a [PolyRingElem], &'a [PolyRingElem]) {
    let (com_rank_1, unif_len, quad_len) = (
        com_params.commit_rank_1,
        com_params.uniform_length,
        com_params.quadratic_length,
    );

    let g_start = unif_len * r * com_rank_1;
    let h_start = g_start + quad_len * r * (r + 1) / 2;

    (
        &tgh.0[..g_start],
        &tgh.0[g_start..h_start],
        &tgh.0[h_start..],
    )
}

pub fn add_apply_matrices_b(
    out: &mut [PolyRingElem],
    matrices_b: &[Vec<PolyMatrix>],
    t: &[PolyRingElem],
    r: usize,
    com_params: &CommitParams,
) {
    let (com_rank_1, unif_len) = (com_params.commit_rank_1, com_params.uniform_length);

    (0..unif_len)
        .into_par_iter()
        .map(|k| {
            (0..r).into_par_iter().map(move |i| {
                let start = (k * r + i) * com_rank_1;
                let end = start + com_rank_1;
                PolyVec(matrices_b[k][i].apply_raw(&t[start..end]))
            })
        })
        .flatten()
        .sum::<PolyVec>()
        .0
        .iter()
        .zip(out.iter_mut())
        .for_each(|(i, o)| *o += i);
}

pub fn add_apply_matrices_garbage(
    out: &mut [PolyRingElem],
    matrices: &[Vec<PolyMatrix>],
    polyvec: &[PolyRingElem],
    r: usize,
    len: usize,
) {
    let choose_2 = |j: usize| j * (j + 1) / 2;

    (0..len)
        .into_par_iter()
        .map(|k| {
            (0..r).into_par_iter().map(move |i| {
                let start = k * choose_2(r) + choose_2(i);
                let end = start + i + 1;
                PolyVec(matrices[k][i].apply_raw(&polyvec[start..end]))
            })
        })
        .flatten()
        .sum::<PolyVec>()
        .0
        .iter()
        .zip(out.iter_mut())
        .for_each(|(i, o)| *o += i);
}
