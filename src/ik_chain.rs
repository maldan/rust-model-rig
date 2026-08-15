//! Configured IK: thin adapter translating `RigDocument`/`BoneId` into
//! `mega_render::ik::IkChainDef` and delegating the actual solve to the
//! render engine.

use glam::Vec3;
use mega_render::{quat_from_matrix, LineOpts, Scene};

use crate::rig::{AppMode, IkChain, RigDocument};

/// Run all enabled chains (Pose).
pub fn evaluate_ik_chains(scene: &mut Scene, rig: &RigDocument) {
    if rig.mode != AppMode::Pose {
        return;
    }
    let chains: Vec<IkChain> = rig.ik_chains.iter().filter(|c| c.enabled).cloned().collect();
    for chain in &chains {
        solve_chain(scene, rig, chain);
    }
}

fn solve_chain(scene: &mut Scene, rig: &RigDocument, chain: &IkChain) {
    if chain.bones.is_empty() {
        return;
    }
    let target_pos = scene
        .world_matrix(chain.target.node)
        .transform_point3(Vec3::ZERO);
    let pole_pos = scene
        .world_matrix(chain.pole.node)
        .transform_point3(Vec3::ZERO);
    let target_rot = {
        let target_world = quat_from_matrix(scene.world_matrix(chain.target.node));
        (target_world * chain.tip_rot_offset).normalize()
    };

    let mut bind_rotations = Vec::with_capacity(chain.bones.len() + 1);
    for &id in chain.bones.iter().chain(std::iter::once(&chain.tip)) {
        let Some(q) = rig.bind_locals.get(&id.node.key()).map(|b| b.rotation) else {
            // Missing bind data for some bone — skip the reset rather than
            // solving from a partially-restored pose.
            bind_rotations.clear();
            break;
        };
        bind_rotations.push(q);
    }

    let def = mega_render::IkChainDef {
        tip: chain.tip.node,
        bones: chain.bones.iter().map(|b| b.node).collect(),
        lengths: chain.lengths.clone(),
        target_pos,
        target_rot,
        pole_pos,
        pole_angle: chain.pole_angle.to_radians(),
        pole_ref_local: chain.pole_ref_local,
        bind_rotations: (!bind_rotations.is_empty()).then_some(bind_rotations),
    };
    mega_render::solve_ik(scene, &def);
}

pub fn draw_ik_helpers(scene: &mut Scene, rig: &RigDocument) {
    if !rig.show_skeleton {
        return;
    }
    for chain in &rig.ik_chains {
        if !chain.enabled || chain.bones.is_empty() {
            continue;
        }
        let tip = scene
            .world_matrix(chain.tip.node)
            .transform_point3(Vec3::ZERO);
        let target = scene
            .world_matrix(chain.target.node)
            .transform_point3(Vec3::ZERO);
        let mid_id = chain.bones[chain.bones.len() / 2];
        let mid = scene.world_matrix(mid_id.node).transform_point3(Vec3::ZERO);
        let pole = scene
            .world_matrix(chain.pole.node)
            .transform_point3(Vec3::ZERO);

        scene.debug.line(
            tip,
            target,
            LineOpts::color([1.0, 0.55, 0.15, 0.85])
                .width(2.0)
                .overlay(),
        );
        scene.debug.line(
            mid,
            pole,
            LineOpts::color([0.85, 0.35, 0.95, 0.85])
                .width(2.0)
                .overlay(),
        );
    }
}