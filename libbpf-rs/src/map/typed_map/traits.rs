//! The capabities that a [`TypedMap`](super::TypedMap) can have.

use crate::MapType;

macro_rules! declare_marker_types {
    (
        $(
            $(#[$docs:meta])*
            $type:ident
                $(== [$map_types0:pat])?
                $(!= [$map_types1:pat])?
            ;
        )*
    ) => {
        $(
            #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
            $(#[$docs])*
            pub enum $type {}
            impl sealed::Sealed for $type {
                fn allows(map_ty: crate::MapType) -> bool {
                    $(::core::matches!(map_ty, $map_types0))*
                    $(!::core::matches!(map_ty, $map_types1))*
                }
            }
        )*
    };
}

mod sealed {
    use crate::MapType;

    pub trait Sealed {
        fn allows(map_ty: MapType) -> bool;
    }
}

/// The trait that defines whether this map is shared by all CPU's in the machine or whether
/// there is a per-cpu instance of the map.
///
/// The possible values for this are [`Shared`] and [`PerCpu`].
pub trait CpuSharing: sealed::Sealed {}

/// The trait that defines whether this map has any built in synchronization primitives.
///
/// The possible valus for this are [`UnSynchronized`], [`Spinlock`] and [`Atomic`].
pub trait Synchronization: sealed::Sealed {}

declare_marker_types! {
    /// Expresses that this map is shared by all cpus.
    Shared != [MapType::PercpuHash | MapType::PercpuArray | MapType::LruPercpuHash | MapType::PercpuCgroupStorage];
    /// Expresses that this map is a per-cpu map.
    PerCpu == [MapType::PercpuHash | MapType::PercpuArray | MapType::LruPercpuHash | MapType::PercpuCgroupStorage];
    /// Expresses that this map doesn't have any synchronization mechanism.
    UnSynchronized == [_]; // any map can be unsynchronized.
    /// Expresses that this map's values have a spin_lock that should be used for synchronization.
    Spinlock == [
        MapType::Hash
            | MapType::PercpuHash
            | MapType::LruHash
            | MapType::LruPercpuHash
            | MapType::Array
            | MapType::PercpuArray
            | MapType::CgroupArray
            | MapType::PercpuCgroupStorage
            | MapType::CgroupStorage
    ];
    /// Expresses that this map provides atomic operations.
    Atomic == [
        MapType::Hash
            | MapType::LruHash
            | MapType::PercpuHash
            | MapType::LruPercpuHash
            | MapType::LpmTrie
    ];
}

impl CpuSharing for Shared {}

impl CpuSharing for PerCpu {}

impl Synchronization for UnSynchronized {}

impl Synchronization for Spinlock {}

impl Synchronization for Atomic {}

#[cfg(test)]
mod test {
    use super::{sealed::Sealed, *};
    use crate::MapType;

    const PER_CPU_TYPES: [MapType; 4] = [
        MapType::PercpuHash,
        MapType::PercpuCgroupStorage,
        MapType::PercpuArray,
        MapType::LruPercpuHash,
    ];

    #[test]
    fn shared_disallows_percpu_types() {
        for ty in PER_CPU_TYPES {
            assert!(!Shared::allows(ty));
        }
    }

    #[test]
    fn per_cpu_allows_percpu_types() {
        for ty in PER_CPU_TYPES {
            assert!(PerCpu::allows(ty));
        }
    }
}
