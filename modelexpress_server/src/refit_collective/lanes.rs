// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lane layout and rank assignment for the NCCL M2N collective path.
//!
//! This is the whole of what MX knows about the transfer's shape. It derives a
//! lane set and a rank per participant from three inputs -- role, ordinal
//! within the role, and source partition -- and nothing else. Tensor, expert,
//! data and pipeline parallelism stay entirely client-side; a change to any of
//! them reaches this module only as a different partition count or a different
//! participant count.
//!
//! Keeping it that way is deliberate. The moment the server has to interpret a
//! parallelism layout to place a rank, every trainer framework needs server
//! support before it can use this path.

use modelexpress_common::grpc::refit_collective::{CollectiveRole, LaneKind};

/// A participant's placement in one lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Assignment {
    pub lane_id: u32,
    pub kind: LaneKind,
    pub rank_in_lane: u32,
    pub world_size: u32,
}

/// The lane set implied by a group's declared membership.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneLayout {
    pub source_partition_count: u32,
    pub trainer_count: u32,
    pub generator_count: u32,
}

/// Lane id of the single broadcast lane. Reshard lanes take `0..partitions`,
/// so the broadcast lane sits immediately after them.
#[must_use]
pub fn broadcast_lane_id(source_partition_count: u32) -> u32 {
    source_partition_count
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LaneError {
    #[error("source_partition_count must be greater than zero")]
    NoPartitions,
    #[error("expected_trainer_slots must not be empty")]
    NoTrainers,
    #[error("expected_generator_slots must not be empty")]
    NoGenerators,
    #[error(
        "trainer count {trainers} is not divisible by source_partition_count {partitions}; \
         every partition must hold the same number of trainer ranks"
    )]
    UnevenPartitions { trainers: u32, partitions: u32 },
    #[error("index_in_role {index} is out of range for {count} {role} slots")]
    IndexOutOfRange {
        index: u32,
        count: u32,
        role: &'static str,
    },
    #[error("a trainer must declare its source_partition")]
    MissingPartition,
    #[error("source_partition {partition} is out of range for {count} partitions")]
    PartitionOutOfRange { partition: u32, count: u32 },
    #[error(
        "trainer index_in_role {index} implies source partition {implied}, not the declared {declared}"
    )]
    PartitionMismatch {
        index: u32,
        implied: u32,
        declared: u32,
    },
    #[error("a generator must not declare a source_partition; it joins every reshard lane")]
    UnexpectedPartition,
    #[error("role must be specified")]
    UnspecifiedRole,
}

impl LaneLayout {
    /// Validate the declared membership and derive the lane geometry.
    pub fn new(
        source_partition_count: u32,
        trainer_count: u32,
        generator_count: u32,
    ) -> Result<Self, LaneError> {
        if source_partition_count == 0 {
            return Err(LaneError::NoPartitions);
        }
        if trainer_count == 0 {
            return Err(LaneError::NoTrainers);
        }
        if generator_count == 0 {
            return Err(LaneError::NoGenerators);
        }
        if !trainer_count.is_multiple_of(source_partition_count) {
            return Err(LaneError::UnevenPartitions {
                trainers: trainer_count,
                partitions: source_partition_count,
            });
        }
        Ok(Self {
            source_partition_count,
            trainer_count,
            generator_count,
        })
    }

    /// Trainer ranks per source partition.
    #[must_use]
    pub fn trainers_per_partition(&self) -> u32 {
        // Non-zero by construction; `new` rejects a zero partition count.
        self.trainer_count
            .checked_div(self.source_partition_count)
            .unwrap_or(0)
    }

    /// World size of one reshard lane: its partition's trainers, plus every
    /// admitted generator. Generators join every reshard lane because a
    /// generator holds all layers and so needs bytes from every partition.
    #[must_use]
    pub fn reshard_world_size(&self) -> u32 {
        self.trainers_per_partition()
            .saturating_add(self.generator_count)
    }

    /// World size of the broadcast lane: every admitted rank on both sides.
    #[must_use]
    pub fn broadcast_world_size(&self) -> u32 {
        self.trainer_count.saturating_add(self.generator_count)
    }

    #[must_use]
    pub fn broadcast_lane_id(&self) -> u32 {
        broadcast_lane_id(self.source_partition_count)
    }

    /// Total lanes: one per source partition, plus the broadcast lane.
    #[must_use]
    pub fn lane_count(&self) -> u32 {
        self.source_partition_count.saturating_add(1)
    }

    /// Every lane this participant belongs to, with its rank in each.
    ///
    /// Trainers occupy the low ranks of a lane and generators follow, so each
    /// lane is a self-contained world whose source ranks are `[0, trainers)`.
    /// The client-side mesh arithmetic depends on exactly that, which is why
    /// the rule lives here rather than being negotiated per deployment.
    pub fn assign(
        &self,
        role: CollectiveRole,
        index_in_role: u32,
        source_partition: Option<u32>,
    ) -> Result<Vec<Assignment>, LaneError> {
        match role {
            CollectiveRole::Trainer => self.assign_trainer(index_in_role, source_partition),
            CollectiveRole::Generator => self.assign_generator(index_in_role, source_partition),
            CollectiveRole::Unspecified => Err(LaneError::UnspecifiedRole),
        }
    }

    fn assign_trainer(
        &self,
        index_in_role: u32,
        source_partition: Option<u32>,
    ) -> Result<Vec<Assignment>, LaneError> {
        if index_in_role >= self.trainer_count {
            return Err(LaneError::IndexOutOfRange {
                index: index_in_role,
                count: self.trainer_count,
                role: "trainer",
            });
        }
        let declared = source_partition.ok_or(LaneError::MissingPartition)?;
        if declared >= self.source_partition_count {
            return Err(LaneError::PartitionOutOfRange {
                partition: declared,
                count: self.source_partition_count,
            });
        }
        // Trainer ordinals are contiguous per partition, so the partition is
        // recoverable from the ordinal. Cross-checking the two catches a
        // client that numbers its ranks one way and its partitions another --
        // which would otherwise place a rank in the wrong lane and hang it.
        let implied = index_in_role
            .checked_div(self.trainers_per_partition())
            .unwrap_or(0);
        if implied != declared {
            return Err(LaneError::PartitionMismatch {
                index: index_in_role,
                implied,
                declared,
            });
        }

        Ok(vec![
            Assignment {
                lane_id: declared,
                kind: LaneKind::Reshard,
                rank_in_lane: index_in_role
                    .checked_rem(self.trainers_per_partition())
                    .unwrap_or(0),
                world_size: self.reshard_world_size(),
            },
            Assignment {
                lane_id: self.broadcast_lane_id(),
                kind: LaneKind::Broadcast,
                rank_in_lane: index_in_role,
                world_size: self.broadcast_world_size(),
            },
        ])
    }

    fn assign_generator(
        &self,
        index_in_role: u32,
        source_partition: Option<u32>,
    ) -> Result<Vec<Assignment>, LaneError> {
        if source_partition.is_some() {
            return Err(LaneError::UnexpectedPartition);
        }
        if index_in_role >= self.generator_count {
            return Err(LaneError::IndexOutOfRange {
                index: index_in_role,
                count: self.generator_count,
                role: "generator",
            });
        }
        let trainers = self.trainers_per_partition();
        let mut assignments: Vec<Assignment> = (0..self.source_partition_count)
            .map(|lane_id| Assignment {
                lane_id,
                kind: LaneKind::Reshard,
                rank_in_lane: trainers.saturating_add(index_in_role),
                world_size: self.reshard_world_size(),
            })
            .collect();
        assignments.push(Assignment {
            lane_id: self.broadcast_lane_id(),
            kind: LaneKind::Broadcast,
            rank_in_lane: self.trainer_count.saturating_add(index_in_role),
            world_size: self.broadcast_world_size(),
        });
        Ok(assignments)
    }

    /// Whether this participant owes a lane its `ncclUniqueId`. Rank 0 of every
    /// reshard lane is a trainer by construction, so only trainers ever do.
    #[must_use]
    pub fn is_bootstrap_leader(&self, role: CollectiveRole, index_in_role: u32) -> bool {
        role == CollectiveRole::Trainer
            && index_in_role
                .checked_rem(self.trainers_per_partition())
                .is_some_and(|remainder| remainder == 0)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn layout(partitions: u32, trainers: u32, generators: u32) -> LaneLayout {
        LaneLayout::new(partitions, trainers, generators).expect("valid layout")
    }

    #[test]
    fn single_partition_puts_trainers_first_and_generators_after() {
        let l = layout(1, 2, 4);
        assert_eq!(l.reshard_world_size(), 6);
        assert_eq!(l.broadcast_world_size(), 6);

        let t0 = l.assign(CollectiveRole::Trainer, 0, Some(0)).expect("t0");
        assert_eq!(t0[0].rank_in_lane, 0);
        let t1 = l.assign(CollectiveRole::Trainer, 1, Some(0)).expect("t1");
        assert_eq!(t1[0].rank_in_lane, 1);
        let g0 = l.assign(CollectiveRole::Generator, 0, None).expect("g0");
        assert_eq!(g0[0].rank_in_lane, 2);
        let g3 = l.assign(CollectiveRole::Generator, 3, None).expect("g3");
        assert_eq!(g3[0].rank_in_lane, 5);
    }

    #[test]
    fn each_reshard_lane_is_a_self_contained_world() {
        // 2 partitions x 2 trainers, 4 generators: each lane is 2 + 4 = 6.
        let l = layout(2, 4, 4);
        assert_eq!(l.trainers_per_partition(), 2);
        assert_eq!(l.reshard_world_size(), 6);
        assert_eq!(l.broadcast_world_size(), 8);

        // Trainer 2 is the first rank of partition 1, so it is rank 0 there.
        let t2 = l.assign(CollectiveRole::Trainer, 2, Some(1)).expect("t2");
        assert_eq!(t2[0].lane_id, 1);
        assert_eq!(t2[0].rank_in_lane, 0);
        // ... but keeps its global ordinal on the broadcast lane.
        assert_eq!(t2[1].kind, LaneKind::Broadcast);
        assert_eq!(t2[1].rank_in_lane, 2);
    }

    #[test]
    fn generators_join_every_reshard_lane_at_the_same_rank() {
        let l = layout(3, 6, 2);
        let g1 = l.assign(CollectiveRole::Generator, 1, None).expect("g1");
        let reshard: Vec<_> = g1.iter().filter(|a| a.kind == LaneKind::Reshard).collect();
        assert_eq!(reshard.len(), 3);
        for (expected_lane, a) in reshard.iter().enumerate() {
            assert_eq!(a.lane_id, u32::try_from(expected_lane).expect("small"));
            // trainers_per_partition (2) + index_in_role (1)
            assert_eq!(a.rank_in_lane, 3);
        }
    }

    #[test]
    fn every_lane_rank_is_assigned_exactly_once() {
        let l = layout(2, 4, 4);
        for lane in 0..l.lane_count() {
            let mut seen: Vec<u32> = Vec::new();
            for i in 0..l.trainer_count {
                for a in l
                    .assign(
                        CollectiveRole::Trainer,
                        i,
                        i.checked_div(l.trainers_per_partition()),
                    )
                    .expect("trainer")
                {
                    if a.lane_id == lane {
                        seen.push(a.rank_in_lane);
                    }
                }
            }
            for i in 0..l.generator_count {
                for a in l.assign(CollectiveRole::Generator, i, None).expect("gen") {
                    if a.lane_id == lane {
                        seen.push(a.rank_in_lane);
                    }
                }
            }
            seen.sort_unstable();
            let world = if lane == l.broadcast_lane_id() {
                l.broadcast_world_size()
            } else {
                l.reshard_world_size()
            };
            let expected: Vec<u32> = (0..world).collect();
            assert_eq!(seen, expected, "lane {lane} must be covered exactly once");
        }
    }

    #[test]
    fn rank_zero_of_every_reshard_lane_is_a_trainer() {
        let l = layout(4, 8, 3);
        for partition in 0_u32..4 {
            let first = partition.saturating_mul(l.trainers_per_partition());
            let a = l
                .assign(CollectiveRole::Trainer, first, Some(partition))
                .expect("trainer");
            assert_eq!(a[0].rank_in_lane, 0);
            assert!(l.is_bootstrap_leader(CollectiveRole::Trainer, first));
        }
        assert!(!l.is_bootstrap_leader(CollectiveRole::Generator, 0));
        assert!(!l.is_bootstrap_leader(CollectiveRole::Trainer, 1));
    }

    #[test]
    fn uneven_partitions_are_rejected() {
        assert_eq!(
            LaneLayout::new(3, 8, 2),
            Err(LaneError::UnevenPartitions {
                trainers: 8,
                partitions: 3
            })
        );
    }

    #[test]
    fn empty_membership_is_rejected() {
        assert_eq!(LaneLayout::new(0, 4, 2), Err(LaneError::NoPartitions));
        assert_eq!(LaneLayout::new(1, 0, 2), Err(LaneError::NoTrainers));
        assert_eq!(LaneLayout::new(1, 4, 0), Err(LaneError::NoGenerators));
    }

    #[test]
    fn a_trainer_numbering_its_partitions_inconsistently_is_rejected() {
        let l = layout(2, 4, 2);
        // Trainer 0 belongs to partition 0; claiming partition 1 would place it
        // in the wrong lane and hang that lane instead of failing.
        assert_eq!(
            l.assign(CollectiveRole::Trainer, 0, Some(1)),
            Err(LaneError::PartitionMismatch {
                index: 0,
                implied: 0,
                declared: 1
            })
        );
    }

    #[test]
    fn role_shaped_misuse_is_rejected() {
        let l = layout(2, 4, 2);
        assert_eq!(
            l.assign(CollectiveRole::Trainer, 0, None),
            Err(LaneError::MissingPartition)
        );
        assert_eq!(
            l.assign(CollectiveRole::Generator, 0, Some(0)),
            Err(LaneError::UnexpectedPartition)
        );
        assert_eq!(
            l.assign(CollectiveRole::Unspecified, 0, None),
            Err(LaneError::UnspecifiedRole)
        );
    }

    #[test]
    fn out_of_range_ordinals_are_rejected() {
        let l = layout(2, 4, 2);
        assert_eq!(
            l.assign(CollectiveRole::Trainer, 4, Some(1)),
            Err(LaneError::IndexOutOfRange {
                index: 4,
                count: 4,
                role: "trainer"
            })
        );
        assert_eq!(
            l.assign(CollectiveRole::Generator, 2, None),
            Err(LaneError::IndexOutOfRange {
                index: 2,
                count: 2,
                role: "generator"
            })
        );
        assert_eq!(
            l.assign(CollectiveRole::Trainer, 0, Some(9)),
            Err(LaneError::PartitionOutOfRange {
                partition: 9,
                count: 2
            })
        );
    }
}
