//! A wrapper for [`Map`](super::Map) that allows you to have more precise borrowing semantics.
//!
//! For example, if your dealing with a [`MapType::Hash`], you don't actually need an exclusive
//! mutable reference to the map to mutate it, this is because the bpf documentation guarentees
//! that operations on this map are atomic. Hence you can convert a [`Map`](super::Map) into a
//! [`TypedMap`]<[`Atomic`](traits::Atomic), [`Shared`](traits::Shared)>. Which has an update method that takes
//! `&self` instead of `&mut self`.

pub mod traits;

use std::marker::PhantomData;
use std::{fmt::Debug, path::Path};

use crate::{Error, Map, MapFlags, MapType, Result};

use ref_cast::RefCast;
use traits::{CpuSharing, Synchronization};

pub mod aliases {
    //! Module with some utility type aliases for commonly used bpf map types
    use super::{traits::*, TypedMap};

    macro_rules! new_type {
        (
            $(#[$docs:meta])*
            $name:ident ($sync:ty, $cpu:ty);
        ) => {
            new_type! {
                $(#[$docs])*
                $name ($sync, $cpu) :: MapType::$name;
            }
        };
        (
            $(#[$docs:meta])*
            $name:ident ($sync:ty, $cpu:ty) :: MapType::$map_ty:ident;
        ) => {
            $(#[$docs])*
            #[derive(Debug, ::ref_cast::RefCast)]
            #[repr(transparent)]
            pub struct $name(super::TypedMap<$sync, $cpu>);

            impl<'map> TryFrom<&'map $crate::Map> for &'map $name {
                type Error = $crate::Error;
                fn try_from(map: &'map $crate::Map) -> $crate::Result<Self> {
                    use ::ref_cast::RefCast;
                    if map.ty == $crate::MapType::$map_ty {
                        Ok( $name::ref_cast(TryInto::try_into(map)?))
                    } else {
                        Err($crate::Error::InvalidInput(format!(
                            "expected map of type {} but got map of type {}",
                            ::core::stringify!($name),
                            map.ty,
                        )))
                    }
                }
            }

            impl<'map> TryFrom<&'map mut $crate::Map> for &'map mut $name {
                type Error = $crate::Error;
                fn try_from(map: &'map mut $crate::Map) -> $crate::Result<Self> {
                    if map.ty == $crate::MapType::$map_ty {
                        use ::ref_cast::RefCast;
                        Ok($name::ref_cast_mut(TryInto::try_into(map)?))
                    } else {
                        Err($crate::Error::InvalidInput(format!(
                            "expected map of type {} but got map of type {}",
                            ::core::stringify!($name),
                            map.ty,
                        )))
                    }
                }
            }

            impl ::core::ops::Deref for $name {
                type Target = TypedMap<$sync, $cpu>;

                fn deref(&self) -> &Self::Target {
                    &self.0
                }
            }

            impl ::core::ops::DerefMut for $name {
                fn deref_mut(&mut self) -> &mut Self::Target {
                    &mut self.0
                }
            }
        };
    }

    new_type!(
        /// Alias to a typed map that represents a hashmap.
        Hash(Atomic, Shared);
    );
    new_type!(
        /// Alias to a typed map that represents a per-cpu hashmap.
        PercpuHash(Atomic, PerCpu);
    );
    new_type!(
        /// Alias to a typed map that represents an array.
        Array(UnSynchronized, Shared);
    );
    new_type!(
        /// Alias to a typed map that represents a per-cpu array.
        PercpuArray(UnSynchronized, PerCpu);
    );
    new_type!(
        /// Alias to a typed map that represents an array that has a spin_lock in each value.
        SpinLockedArray(Spinlock, Shared) :: MapType::Array;
    );
    new_type!(
        /// Alias to a typed map that represents a per-cpu array that has a spin_lock in each value.
        SpinLockedPerCpuArray(Spinlock, PerCpu) :: MapType::PercpuArray;
    );
}

/// Represents a created map.
///
/// Some methods require working with raw bytes. You may find libraries such as
/// [`plain`](https://crates.io/crates/plain) helpful.
#[derive(Debug, RefCast)]
#[repr(transparent)]
pub struct TypedMap<S, C> {
    erased_map: Map,
    _map_properties: PhantomData<(S, C)>,
}

impl<'map, S: Synchronization, C: CpuSharing> TryFrom<&'map Map> for &'map TypedMap<S, C> {
    type Error = Error;
    fn try_from(erased: &'map Map) -> Result<Self> {
        try_from_erased_map_checks::<S, C>(erased)?;
        Ok(TypedMap::ref_cast(erased))
    }
}

impl<'map, S: Synchronization, C: CpuSharing> TryFrom<&'map mut Map> for &'map mut TypedMap<S, C> {
    type Error = Error;
    fn try_from(erased: &'map mut Map) -> Result<Self> {
        try_from_erased_map_checks::<S, C>(erased)?;
        Ok(TypedMap::ref_cast_mut(erased))
    }
}

fn try_from_erased_map_checks<S: Synchronization, C: CpuSharing>(erased: &Map) -> Result<()> {
    if !S::allows(erased.ty) {
        return Err(Error::InvalidInput(format!(
            "map of type {} can't be {}",
            erased.ty,
            std::any::type_name::<S>()
        )));
    }

    if !C::allows(erased.ty) {
        return Err(Error::InvalidInput(format!(
            "map of type {} can't be {}",
            erased.ty,
            std::any::type_name::<C>()
        )));
    }
    Ok(())
}

macro_rules! copy_api_from_erased {
    ($($name:ident -> $ret:ty;)*) => {
        $(
            #[doc = ::core::concat!("See [`crate::Map::", ::core::stringify!($name), "`]")]
            pub fn $name(&self) -> $ret {
                self.erased_map.$name()
            }
        )*
    };
}

impl<S, C> TypedMap<S, C> {
    copy_api_from_erased! {
        name -> &str;
        fd -> i32;
        map_type -> MapType;
        key_size -> u32;
        value_size -> u32;
    }

    /// See [`crate::Map::pin`].
    pub fn pin<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.erased_map.pin(path)
    }

    /// See [`crate::Map::unpin`].
    pub fn unpin<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.erased_map.unpin(path)
    }
}

impl<C> TypedMap<traits::UnSynchronized, C> {
    /// Deletes an element from the map.
    ///
    /// `key` must have exactly [`TypedMap::key_size()`] elements.
    pub fn delete(&mut self, key: &[u8]) -> Result<()> {
        // SAFETY:
        // we have exclusive access to the map because self is being passed by exclusive reference.
        unsafe { self.erased_map.delete_unchecked(key) }
    }
}

impl<C> TypedMap<traits::Atomic, C> {
    /// Deletes an element from the map.
    ///
    /// `key` must have exactly [`TypedMap::key_size()`] elements.
    pub fn delete(&self, key: &[u8]) -> Result<()> {
        // SAFETY:
        // Hashmaps always support atomic operations
        unsafe { self.erased_map.delete_unchecked(key) }
    }
}

impl<C> TypedMap<traits::Spinlock, C> {
    /// Deletes an element from the map.
    ///
    /// `key` must have exactly [`TypedMap::key_size()`] elements.
    pub fn delete(&mut self, key: &[u8]) -> Result<()> {
        // SAFETY:
        // we have exclusive access to the map because self is being passed by exclusive reference.
        unsafe { self.erased_map.delete_unchecked(key) }
    }
}

impl TypedMap<traits::UnSynchronized, traits::Shared> {
    /// Returns map value as `Vec` of `u8`.
    ///
    /// `key` must have exactly [`TypedMap::key_size()`] elements.
    /// must be used.
    pub fn lookup(&self, key: &[u8], flags: MapFlags) -> Result<Option<Vec<u8>>> {
        let out_size = self.erased_map.value_size() as usize;
        self.erased_map.lookup_raw(key, flags, out_size)
    }

    /// Update an element.
    ///
    /// `key` must have exactly [`TypedMap::key_size()`] elements. `value` must have exactly
    /// [`TypedMap::value_size()`] elements.
    pub fn update(&mut self, key: &[u8], value: &[u8], flags: MapFlags) -> Result<()> {
        self.erased_map.check_value_size(value)?;

        // SAFETY: we have a mutable reference to `self` thus the compiler guarentees we are the
        // only ones looking at this map.
        unsafe { self.erased_map.update_raw(key, value, flags) }
    }
}

impl TypedMap<traits::Atomic, traits::Shared> {
    /// Returns map value as `Vec` of `u8`.
    ///
    /// `key` must have exactly [`TypedMap::key_size()`] elements.
    pub fn lookup(&self, key: &[u8], flags: MapFlags) -> Result<Option<Vec<u8>>> {
        let out_size = self.erased_map.value_size() as usize;
        self.erased_map.lookup_raw(key, flags, out_size)
    }

    /// Equivalent to [`super::Map::update`] but takes self through a shared reference.
    pub fn update(&self, key: &[u8], value: &[u8], flags: MapFlags) -> Result<()> {
        self.erased_map.check_value_size(value)?;
        // SAFETY:
        // - Check value size makes sure the size of the value is equal to value_size
        // - Hashmaps always support atomic operations
        unsafe { self.erased_map.update_raw(key, value, flags) }
    }
}

impl TypedMap<traits::Spinlock, traits::Shared> {
    /// Returns map value as `Vec` of `u8`.
    ///
    /// `key` must have exactly [`TypedMap::key_size()`] elements.
    pub fn lookup(&self, key: &[u8], flags: MapFlags) -> Result<Option<Vec<u8>>> {
        let out_size = self.erased_map.value_size() as usize;
        self.erased_map
            .lookup_raw(key, flags | MapFlags::LOCK, out_size)
    }

    /// Equivalent to [`super::Map::update`] but takes self through a shared reference.
    pub fn update(&self, key: &[u8], value: &[u8], flags: MapFlags) -> Result<()> {
        self.erased_map.check_value_size(value)?;
        // SAFETY:
        // - Check value size makes sure the size of the value is equal to value_size
        // - Hashmaps always support atomic operations
        unsafe {
            self.erased_map
                .update_raw(key, value, flags | MapFlags::LOCK)
        }
    }
}

impl TypedMap<traits::UnSynchronized, traits::PerCpu> {
    /// Returns one value per cpu as `Vec` of `Vec` of `u8` for per per-cpu maps.
    ///
    /// For normal maps, [`TypedMap::lookup()`] must be used.
    pub fn lookup_percpu(&self, key: &[u8], flags: MapFlags) -> Result<Option<Vec<Vec<u8>>>> {
        // SAFETY:
        // This is a per-cpu map, the compiler guarentees it.
        unsafe { self.erased_map.lookup_percpu_unchecked(key, flags) }
    }

    /// Update an element in an per-cpu map with one value per cpu.
    ///
    /// `key` must have exactly [`TypedMap::key_size()`] elements. `value` must have one
    /// element per cpu (see [`crate::num_possible_cpus()`]) with exactly [`TypedMap::value_size()`]
    /// elements each.
    pub fn update_percpu<B>(&mut self, key: &[u8], values: &[B], flags: MapFlags) -> Result<()>
    where
        B: AsRef<[u8]>,
    {
        // SAFETY:
        // - self is being taken by exclusive reference so we have exclusive access to this map
        // - we checked that the number of values matches the number of cpus
        unsafe { self.erased_map.update_percpu_unchecked(key, values, flags) }
    }
}

impl TypedMap<traits::Atomic, traits::PerCpu> {
    /// Returns one value per cpu as `Vec` of `Vec` of `u8` for per per-cpu maps.
    ///
    /// For normal maps, [`TypedMap::lookup()`] must be used.
    pub fn lookup_percpu(&self, key: &[u8], flags: MapFlags) -> Result<Option<Vec<Vec<u8>>>> {
        // SAFETY:
        // This is a per-cpu map, the compiler guarentees it.
        unsafe { self.erased_map.lookup_percpu_unchecked(key, flags) }
    }

    /// Update an element in an per-cpu map with one value per cpu.
    ///
    /// `key` must have exactly [`TypedMap::key_size()`] elements. `value` must have one
    /// element per cpu (see [`crate::num_possible_cpus()`]) with exactly [`TypedMap::value_size()`]
    /// elements each.
    pub fn update_percpu<B>(&self, key: &[u8], values: &[B], flags: MapFlags) -> Result<()>
    where
        B: AsRef<[u8]>,
    {
        // SAFETY:
        // This is a per-cpu map, the compiler guarentees it.
        // This map supports atomic operations.
        unsafe { self.erased_map.update_percpu_unchecked(key, values, flags) }
    }
}

impl TypedMap<traits::Spinlock, traits::PerCpu> {
    /// Returns one value per cpu as `Vec` of `Vec` of `u8` for per per-cpu maps.
    ///
    /// For normal maps, [`TypedMap::lookup()`] must be used.
    pub fn lookup_percpu(&self, key: &[u8], flags: MapFlags) -> Result<Option<Vec<Vec<u8>>>> {
        // SAFETY:
        // This is a per-cpu map, the compiler guarentees it.
        // This map supports spinlocks operations.
        unsafe {
            self.erased_map
                .lookup_percpu_unchecked(key, flags | MapFlags::LOCK)
        }
    }

    /// Update an element in an per-cpu map with one value per cpu.
    ///
    /// `key` must have exactly [`TypedMap::key_size()`] elements. `value` must have one
    /// element per cpu (see [`crate::num_possible_cpus()`]) with exactly [`TypedMap::value_size()`]
    /// elements each.
    pub fn update_percpu<B>(&self, key: &[u8], values: &[B], flags: MapFlags) -> Result<()>
    where
        B: AsRef<[u8]>,
    {
        // SAFETY:
        // This is a per-cpu map, the compiler guarentees it.
        // This map supports spinlocks operations.
        unsafe {
            self.erased_map
                .update_percpu_unchecked(key, values, flags | MapFlags::LOCK)
        }
    }
}
