// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lane membership and rank assignment for the NCCL M2N collective path.
//!
//! This is the whole of what MX knows about the transfer's shape, and it is
//! only what the caller declared: a set of lanes, each an ordered list of
//! slots. MX brokers one bootstrap per lane, assigns each slot the rank its
//! position gives it, and gates readiness on every lane. It does not derive the
//! lane set, does not infer which lanes a slot belongs to, and has no notion of
//! a source partition.
//!
//! Keeping it that way is the point. Tensor, expert, data and pipeline
//! parallelism are the caller's; the moment the server has to interpret a
//! parallelism layout to place a rank, every trainer framework needs server
//! support before it can use this path.

use std::collections::{HashMap, HashSet};

use modelexpress_common::grpc::refit_collective::LaneKind;

/// A participant's placement in one lane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assignment {
    pub lane_id: u32,
    pub kind: LaneKind,
    pub rank_in_lane: u32,
    pub world_size: u32,
}

/// One communicator, exactly as the caller declared it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lane {
    pub lane_id: u32,
    pub kind: LaneKind,
    /// Slots in rank order. Trainers take the low ranks, generators follow, so
    /// a lane's source ranks are always `[0, trainer_slots.len())`.
    pub trainer_slots: Vec<String>,
    pub generator_slots: Vec<String>,
}

impl Lane {
    fn slots_in_rank_order(&self) -> impl Iterator<Item = &String> {
        self.trainer_slots.iter().chain(self.generator_slots.iter())
    }

    #[must_use]
    pub fn world_size(&self) -> u32 {
        let total = self.trainer_slots.len().saturating_add(self.generator_slots.len());
        u32::try_from(total).unwrap_or(u32::MAX)
    }
}

/// The declared lanes of one operation, indexed for lookup.
#[derive(Debug, Clone)]
pub struct LaneLayout {
    lanes: Vec<Lane>,
    /// slot_id -> the assignments that slot holds, in declared lane order.
    by_slot: HashMap<String, Vec<Assignment>>,
    pub trainer_count: u32,
    pub generator_count: u32,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LaneError {
    #[error("a collective group must declare at least one lane")]
    NoLanes,
    #[error("expected_trainer_slots must not be empty")]
    NoTrainers,
    #[error("expected_generator_slots must not be empty")]
    NoGenerators,
    #[error("lane_id {lane_id} is declared more than once")]
    DuplicateLaneId { lane_id: u32 },
    #[error("lane {lane_id} declares kind LANE_KIND_UNSPECIFIED")]
    UnspecifiedLaneKind { lane_id: u32 },
    #[error("lane {lane_id} declares slot {slot_id}, which is not an expected {role} slot")]
    UnknownSlot {
        lane_id: u32,
        slot_id: String,
        role: &'static str,
    },
    #[error("lane {lane_id} declares slot {slot_id} more than once")]
    DuplicateSlotInLane { lane_id: u32, slot_id: String },
    #[error("slot {slot_id} is expected but is on no declared lane")]
    UnassignedSlot { slot_id: String },
    #[error("at most one broadcast lane may be declared; found {found}")]
    MultipleBroadcastLanes { found: usize },
    #[error("slot {slot_id} is not a member of this group")]
    UnknownParticipant { slot_id: String },
}

impl LaneLayout {
    /// Validate a declared lane set against the group's expected slots.
    ///
    /// Every check here is structural: unique lane ids, slots that exist, no
    /// slot declared twice on one lane, no expected slot left off every lane.
    /// None of them is a claim about what the lanes mean.
    pub fn new(
        lanes: Vec<Lane>,
        expected_trainer_slots: &[String],
        expected_generator_slots: &[String],
    ) -> Result<Self, LaneError> {
        if expected_trainer_slots.is_empty() {
            return Err(LaneError::NoTrainers);
        }
        if expected_generator_slots.is_empty() {
            return Err(LaneError::NoGenerators);
        }
        if lanes.is_empty() {
            return Err(LaneError::NoLanes);
        }

        let trainers: HashSet<&String> = expected_trainer_slots.iter().collect();
        let generators: HashSet<&String> = expected_generator_slots.iter().collect();

        let broadcast = lanes
            .iter()
            .filter(|lane| lane.kind == LaneKind::Broadcast)
            .count();
        if broadcast > 1 {
            return Err(LaneError::MultipleBroadcastLanes { found: broadcast });
        }

        let mut seen_lane_ids: HashSet<u32> = HashSet::new();
        let mut by_slot: HashMap<String, Vec<Assignment>> = HashMap::new();

        for lane in &lanes {
            if !seen_lane_ids.insert(lane.lane_id) {
                return Err(LaneError::DuplicateLaneId {
                    lane_id: lane.lane_id,
                });
            }
            if lane.kind == LaneKind::Unspecified {
                return Err(LaneError::UnspecifiedLaneKind {
                    lane_id: lane.lane_id,
                });
            }
            for (slot_id, known, role) in lane
                .trainer_slots
                .iter()
                .map(|s| (s, &trainers, "trainer"))
                .chain(
                    lane.generator_slots
                        .iter()
                        .map(|s| (s, &generators, "generator")),
                )
            {
                if !known.contains(slot_id) {
                    return Err(LaneError::UnknownSlot {
                        lane_id: lane.lane_id,
                        slot_id: slot_id.clone(),
                        role,
                    });
                }
            }

            let world_size = lane.world_size();
            let mut seen_in_lane: HashSet<&String> = HashSet::new();
            for (rank, slot_id) in lane.slots_in_rank_order().enumerate() {
                if !seen_in_lane.insert(slot_id) {
                    return Err(LaneError::DuplicateSlotInLane {
                        lane_id: lane.lane_id,
                        slot_id: slot_id.clone(),
                    });
                }
                by_slot.entry(slot_id.clone()).or_default().push(Assignment {
                    lane_id: lane.lane_id,
                    kind: lane.kind,
                    rank_in_lane: u32::try_from(rank).unwrap_or(u32::MAX),
                    world_size,
                });
            }
        }

        // A slot on no lane would be admitted, counted toward readiness, and
        // then wait on a communicator it was never placed in.
        for slot_id in expected_trainer_slots.iter().chain(expected_generator_slots) {
            if !by_slot.contains_key(slot_id) {
                return Err(LaneError::UnassignedSlot {
                    slot_id: slot_id.clone(),
                });
            }
        }

        Ok(Self {
            lanes,
            by_slot,
            trainer_count: u32::try_from(expected_trainer_slots.len()).unwrap_or(u32::MAX),
            generator_count: u32::try_from(expected_generator_slots.len()).unwrap_or(u32::MAX),
        })
    }

    #[must_use]
    pub fn lanes(&self) -> &[Lane] {
        &self.lanes
    }

    #[must_use]
    pub fn lane_count(&self) -> u32 {
        u32::try_from(self.lanes.len()).unwrap_or(u32::MAX)
    }

    #[must_use]
    pub fn lane(&self, lane_id: u32) -> Option<&Lane> {
        self.lanes.iter().find(|lane| lane.lane_id == lane_id)
    }

    /// The lane the caller declared as spanning every participant, if any. MX
    /// reads it only to report which slots have not been admitted yet.
    #[must_use]
    pub fn broadcast_lane_id(&self) -> Option<u32> {
        self.lanes
            .iter()
            .find(|lane| lane.kind == LaneKind::Broadcast)
            .map(|lane| lane.lane_id)
    }

    /// Where this slot sits in every lane that declared it.
    pub fn assign(&self, slot_id: &str) -> Result<Vec<Assignment>, LaneError> {
        self.by_slot
            .get(slot_id)
            .cloned()
            .ok_or_else(|| LaneError::UnknownParticipant {
                slot_id: slot_id.to_string(),
            })
    }

    /// Whether this slot owes any lane its `ncclUniqueId`. Rank 0 of a lane is
    /// a trainer by construction, since trainers take the low ranks.
    #[must_use]
    pub fn is_bootstrap_leader(&self, slot_id: &str) -> bool {
        self.by_slot
            .get(slot_id)
            .is_some_and(|assignments| assignments.iter().any(|a| a.rank_in_lane == 0))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn s(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| (*v).to_string()).collect()
    }

    /// Two reshard lanes plus a broadcast lane, declared the way a caller with
    /// two pipeline stages would declare them. Nothing here tells MX that.
    fn layout() -> LaneLayout {
        LaneLayout::new(
            vec![
                Lane {
                    lane_id: 0,
                    kind: LaneKind::Reshard,
                    trainer_slots: s(&["t0", "t1"]),
                    generator_slots: s(&["g0", "g1"]),
                },
                Lane {
                    lane_id: 1,
                    kind: LaneKind::Reshard,
                    trainer_slots: s(&["t2", "t3"]),
                    generator_slots: s(&["g0", "g1"]),
                },
                Lane {
                    lane_id: 2,
                    kind: LaneKind::Broadcast,
                    trainer_slots: s(&["t0", "t1", "t2", "t3"]),
                    generator_slots: s(&["g0", "g1"]),
                },
            ],
            &s(&["t0", "t1", "t2", "t3"]),
            &s(&["g0", "g1"]),
        )
        .expect("layout")
    }

    #[test]
    fn a_rank_is_the_slots_position_in_the_lane_it_was_declared_on() {
        let layout = layout();
        assert_eq!(
            layout.assign("t2").expect("t2"),
            vec![
                Assignment {
                    lane_id: 1,
                    kind: LaneKind::Reshard,
                    rank_in_lane: 0,
                    world_size: 4
                },
                Assignment {
                    lane_id: 2,
                    kind: LaneKind::Broadcast,
                    rank_in_lane: 2,
                    world_size: 6
                },
            ]
        );
    }

    #[test]
    fn a_generator_takes_the_ranks_after_the_trainers_on_every_lane_it_is_on() {
        let layout = layout();
        let g1 = layout.assign("g1").expect("g1");
        assert_eq!(g1.len(), 3);
        assert_eq!(g1[0].rank_in_lane, 3);
        assert_eq!(g1[1].rank_in_lane, 3);
        assert_eq!(g1[2].rank_in_lane, 5);
    }

    #[test]
    fn only_a_lanes_rank_zero_owes_it_a_bootstrap() {
        let layout = layout();
        assert!(layout.is_bootstrap_leader("t0"));
        assert!(layout.is_bootstrap_leader("t2"));
        assert!(!layout.is_bootstrap_leader("t1"));
        assert!(!layout.is_bootstrap_leader("g0"));
    }

    #[test]
    fn the_broadcast_lane_is_whichever_one_the_caller_labelled() {
        assert_eq!(layout().broadcast_lane_id(), Some(2));
    }

    #[test]
    fn a_layout_with_no_broadcast_lane_is_valid() {
        let layout = LaneLayout::new(
            vec![Lane {
                lane_id: 7,
                kind: LaneKind::Reshard,
                trainer_slots: s(&["t0"]),
                generator_slots: s(&["g0"]),
            }],
            &s(&["t0"]),
            &s(&["g0"]),
        )
        .expect("layout");
        assert_eq!(layout.broadcast_lane_id(), None);
        assert_eq!(layout.lane_count(), 1);
    }

    #[test]
    fn a_slot_left_off_every_lane_is_rejected_rather_than_admitted_and_hung() {
        let error = LaneLayout::new(
            vec![Lane {
                lane_id: 0,
                kind: LaneKind::Reshard,
                trainer_slots: s(&["t0"]),
                generator_slots: s(&["g0"]),
            }],
            &s(&["t0", "t1"]),
            &s(&["g0"]),
        )
        .expect_err("t1 is on no lane");
        assert_eq!(
            error,
            LaneError::UnassignedSlot {
                slot_id: "t1".to_string()
            }
        );
    }

    #[test]
    fn a_lane_declaring_a_slot_the_group_does_not_have_is_rejected() {
        let error = LaneLayout::new(
            vec![Lane {
                lane_id: 0,
                kind: LaneKind::Reshard,
                trainer_slots: s(&["t0", "ghost"]),
                generator_slots: s(&["g0"]),
            }],
            &s(&["t0"]),
            &s(&["g0"]),
        )
        .expect_err("ghost is not a trainer slot");
        assert!(matches!(error, LaneError::UnknownSlot { .. }));
    }

    #[test]
    fn a_repeated_lane_id_is_rejected() {
        let error = LaneLayout::new(
            vec![
                Lane {
                    lane_id: 0,
                    kind: LaneKind::Reshard,
                    trainer_slots: s(&["t0"]),
                    generator_slots: s(&["g0"]),
                },
                Lane {
                    lane_id: 0,
                    kind: LaneKind::Broadcast,
                    trainer_slots: s(&["t0"]),
                    generator_slots: s(&["g0"]),
                },
            ],
            &s(&["t0"]),
            &s(&["g0"]),
        )
        .expect_err("lane 0 twice");
        assert_eq!(error, LaneError::DuplicateLaneId { lane_id: 0 });
    }

    #[test]
    fn a_slot_declared_twice_on_one_lane_is_rejected() {
        let error = LaneLayout::new(
            vec![Lane {
                lane_id: 0,
                kind: LaneKind::Reshard,
                trainer_slots: s(&["t0", "t0"]),
                generator_slots: s(&["g0"]),
            }],
            &s(&["t0"]),
            &s(&["g0"]),
        )
        .expect_err("t0 twice on lane 0");
        assert!(matches!(error, LaneError::DuplicateSlotInLane { .. }));
    }

    #[test]
    fn two_broadcast_lanes_are_rejected() {
        let error = LaneLayout::new(
            vec![
                Lane {
                    lane_id: 0,
                    kind: LaneKind::Broadcast,
                    trainer_slots: s(&["t0"]),
                    generator_slots: s(&["g0"]),
                },
                Lane {
                    lane_id: 1,
                    kind: LaneKind::Broadcast,
                    trainer_slots: s(&["t0"]),
                    generator_slots: s(&["g0"]),
                },
            ],
            &s(&["t0"]),
            &s(&["g0"]),
        )
        .expect_err("two broadcast lanes");
        assert_eq!(error, LaneError::MultipleBroadcastLanes { found: 2 });
    }

    #[test]
    fn a_slot_that_is_not_a_member_cannot_be_assigned() {
        let error = layout().assign("nobody").expect_err("not a member");
        assert!(matches!(error, LaneError::UnknownParticipant { .. }));
    }

    /// An uneven split is the case the derived layout used to refuse outright.
    /// Nothing about it is MX's business once the caller declares the lanes.
    #[test]
    fn an_uneven_split_is_just_another_declaration() {
        let layout = LaneLayout::new(
            vec![
                Lane {
                    lane_id: 0,
                    kind: LaneKind::Reshard,
                    trainer_slots: s(&["t0"]),
                    generator_slots: s(&["g0"]),
                },
                Lane {
                    lane_id: 1,
                    kind: LaneKind::Reshard,
                    trainer_slots: s(&["t1", "t2"]),
                    generator_slots: s(&["g0"]),
                },
            ],
            &s(&["t0", "t1", "t2"]),
            &s(&["g0"]),
        )
        .expect("uneven is fine");
        assert_eq!(layout.assign("t2").expect("t2")[0].rank_in_lane, 1);
        assert_eq!(layout.lane_count(), 2);
    }
}
