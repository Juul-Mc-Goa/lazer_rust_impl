#[macro_use]
extern crate lazy_static;

mod commit;
mod constants;
mod constraint;
mod dachshund;
mod matrices;
mod proof;
mod recursive_prover;
mod ring;
mod statement;
mod witness;

fn main() {
    use aes::cipher::{KeyIvInit, StreamCipher};
    use witness::Aes128Ctr64LE;

    let key = [0u8; 16];
    let mut cipher = Aes128Ctr64LE::new(&key.into(), &0_u128.to_le_bytes().into());

    let print_buf = |v: Vec<u8>| {
        for i in 0..32 {
            for j in 0..16 {
                print!("{:3} ", v[16 * i + j])
            }
            print!("\n");
        }
        print!("\n");
    };

    for _ in 0..2 {
        let mut buf = vec![0_u8; 512];
        cipher.apply_keystream(&mut buf);

        print_buf(buf);
    }

    // REVIEW: applying keystream twice is NOT like setting iv = 1
    let mut cipher = Aes128Ctr64LE::new(&key.into(), &1_u128.to_le_bytes().into());
    let mut buf = vec![0_u8; 512];
    cipher.apply_keystream(&mut buf);

    print_buf(buf);
}
