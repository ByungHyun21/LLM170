//! [R6 2026-10-10] 런치 인자 부트스트랩 — 커널 인자 배열 조립 헬퍼.
//!
//! 종전 각 발사부의 `let mut args: [*mut c_void; N] = [ (&mut a) as … ]`
//! (43곳)을 `&mut lN(&mut a, &mut b, …)` 한 식으로 축약한다. 동작 동일:
//! 같은 로컬을 대여해 같은 순서의 포인터 배열을 만든다(임시 배열은 launch
//! 호출식 동안만 산다).
//!
//! 계약: 인자는 launch 직전까지 살아 있는 로컬(또는 호출자 소유 필드의
//! 대여)이어야 한다 — 배열이 커널에 전달되는 동안 유효해야 함.

use std::ffi::c_void;

macro_rules! args_fn {
    ($name:ident, $n:expr, $($T:ident $v:ident),+) => {
        // 헬퍼 패밀리 대칭 유지 — 미사용 아리티도 남긴다(다음 사이트 대비).
        #[allow(clippy::too_many_arguments)]
        #[allow(dead_code)]
        pub(crate) fn $name<$($T),+>($($v: &mut $T),+) -> [*mut c_void; $n] {
            [$($v as *mut $T as *mut c_void),+]
        }
    };
}

args_fn!(l1, 1, A a);
args_fn!(l2, 2, A a, B b);
args_fn!(l3, 3, A a, B b, C c);
args_fn!(l4, 4, A a, B b, C c, D d);
args_fn!(l5, 5, A a, B b, C c, D d, E e);
args_fn!(l6, 6, A a, B b, C c, D d, E e, F f);
args_fn!(l7, 7, A a, B b, C c, D d, E e, F f, G g);
args_fn!(l8, 8, A a, B b, C c, D d, E e, F f, G g, H h);
args_fn!(l9, 9, A a, B b, C c, D d, E e, F f, G g, H h, I i);
args_fn!(l10, 10, A a, B b, C c, D d, E e, F f, G g, H h, I i, J j);
args_fn!(l11, 11, A a, B b, C c, D d, E e, F f, G g, H h, I i, J j, K k);
args_fn!(l12, 12, A a, B b, C c, D d, E e, F f, G g, H h, I i, J j, K k, L l);
args_fn!(l13, 13, A a, B b, C c, D d, E e, F f, G g, H h, I i, J j, K k, L l, M m);
args_fn!(l14, 14, A a, B b, C c, D d, E e, F f, G g, H h, I i, J j, K k, L l, M m, N n);
args_fn!(l15, 15, A a, B b, C c, D d, E e, F f, G g, H h, I i, J j, K k, L l, M m, N n, O o);
args_fn!(l16, 16, A a, B b, C c, D d, E e, F f, G g, H h, I i, J j, K k, L l, M m, N n, O o, P p);

#[cfg(test)]
mod tests {
    use super::*;

    /// 순서·주소 계약 — 인자 순서가 배열 순서와 일치.
    #[test]
    fn args_order_and_pointers() {
        let (mut a, mut b, mut c) = (1i32, 2u64, 3.0f32);
        let arr = l3(&mut a, &mut b, &mut c);
        assert_eq!(arr[0] as *mut i32, &mut a as *mut i32);
        assert_eq!(arr[1] as *mut u64, &mut b as *mut u64);
        assert_eq!(arr[2] as *mut f32, &mut c as *mut f32);
    }
}
