#[macro_use]
extern crate lazy_static;

mod constants;
mod dachshund;
mod matrices;
mod recursive_prover;
mod ring;
mod witness;

fn main() {
    use aes::cipher::{KeyIvInit, StreamCipher};
    use witness::Aes128Ctr64LE;

    let hashbuf = [0u8; 16];
    let mut cipher = Aes128Ctr64LE::new(&hashbuf.into(), &0_u128.to_le_bytes().into());

    for _ in 0..8 {
        let mut buf = vec![0_u8; 512];
        cipher.apply_keystream(&mut buf);

        for i in 0..32 {
            for j in 0..16 {
                print!("{:3} ", buf[16 * i + j])
            }
            print!("\n");
        }
        print!("\n");
    }
}
