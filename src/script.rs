//! Lua macros: load a model, wire drivers, create IK / soft / colliders.

use std::path::{Path, PathBuf};
use std::ptr::NonNull;

use glam::Vec2;
use mlua::{Lua, Table, UserData, UserDataFields, UserDataMethods, Value, Variadic};

use crate::app::AppState;
use crate::driver::{DriverNodeKind, DriverSpace};
use crate::rig::{AppMode, BoneId};

struct ScriptEnv {
    app: NonNull<AppState>,
    script_dir: PathBuf,
}

fn env(lua: &Lua) -> mlua::Result<mlua::AppDataRef<'_, ScriptEnv>> {
    lua.app_data_ref::<ScriptEnv>()
        .ok_or_else(|| mlua::Error::runtime("script host is gone"))
}

fn app(lua: &Lua) -> mlua::Result<&'static mut AppState> {
    let env = env(lua)?;
    Ok(unsafe { &mut *env.app.as_ptr() })
}

fn lua_err(msg: impl Into<String>) -> mlua::Error {
    mlua::Error::runtime(msg.into())
}

fn resolve_path(lua: &Lua, path: &str) -> mlua::Result<PathBuf> {
    let p = PathBuf::from(path);
    if p.is_absolute() {
        return Ok(p);
    }
    Ok(env(lua)?.script_dir.join(p))
}

fn require_bone(state: &AppState, name: &str) -> mlua::Result<BoneId> {
    state
        .rig
        .bone_by_name(name)
        .ok_or_else(|| lua_err(format!("unknown bone '{name}' (rig.list_bones())")))
}

fn glob_match(pat: &str, s: &str) -> bool {
    fn rec(p: &[u8], s: &[u8]) -> bool {
        match p.split_first() {
            None => s.is_empty(),
            Some((b'*', rest)) => {
                rec(rest, s) || s.split_first().is_some_and(|(_, tail)| rec(p, tail))
            }
            Some((b'?', rest)) => s.split_first().is_some_and(|(_, tail)| rec(rest, tail)),
            Some((c, rest)) => s
                .split_first()
                .is_some_and(|(sc, tail)| sc == c && rec(rest, tail)),
        }
    }
    rec(pat.as_bytes(), s.as_bytes())
}

fn table_f32(opts: &Table, key: &str) -> mlua::Result<Option<f32>> {
    match opts.get::<Value>(key)? {
        Value::Nil => Ok(None),
        Value::Number(n) => Ok(Some(n as f32)),
        Value::Integer(n) => Ok(Some(n as f32)),
        other => Err(lua_err(format!(
            "option '{key}' must be a number, got {other:?}"
        ))),
    }
}

fn table_vec2(opts: &Table, key: &str) -> mlua::Result<Option<Vec2>> {
    match opts.get::<Value>(key)? {
        Value::Nil => Ok(None),
        Value::Table(t) => {
            let x = t
                .get::<Option<f32>>("x")?
                .or(t.get::<Option<f32>>(1)?)
                .unwrap_or(0.0);
            let y = t
                .get::<Option<f32>>("y")?
                .or(t.get::<Option<f32>>(2)?)
                .unwrap_or(0.0);
            Ok(Some(Vec2::new(x, y)))
        }
        _ => Err(lua_err(format!("option '{key}' must be {{x, y}}"))),
    }
}

fn apply_node_opts(
    state: &mut AppState,
    driver_id: u32,
    node_id: &str,
    opts: &Table,
) -> mlua::Result<()> {
    let bone = match opts.get::<Option<String>>("bone")? {
        Some(name) => Some(require_bone(state, &name)?),
        None => None,
    };
    let space = match opts.get::<Option<String>>("space")? {
        Some(s) => Some(
            DriverSpace::from_name(&s)
                .ok_or_else(|| lua_err(format!("unknown space '{s}' (local|world|offset)")))?,
        ),
        None => None,
    };

    let driver = state
        .rig
        .driver_mut(driver_id)
        .ok_or_else(|| lua_err(format!("unknown driver {driver_id}")))?;
    let node = driver
        .node_mut(node_id)
        .ok_or_else(|| lua_err(format!("unknown node '{node_id}'")))?;
    if let Some(bone) = bone {
        node.bone = Some(bone);
    }
    if let Some(space) = space {
        node.space = space;
    }
    apply_node_floats(node, opts)
}

fn apply_node_floats(node: &mut crate::driver::DriverNode, opts: &Table) -> mlua::Result<()> {
    if let Some(v) = table_f32(opts, "value")? {
        node.floats[0] = v;
    }
    if let Some(v) = table_f32(opts, "t")? {
        node.floats[0] = v;
    }
    if let Some(v) = table_f32(opts, "from")? {
        node.floats[0] = v;
    }
    if let Some(v) = table_f32(opts, "to")? {
        node.floats[1] = v;
    }
    if let Some(v) = table_f32(opts, "in_from")? {
        node.floats[0] = v;
    }
    if let Some(v) = table_f32(opts, "in_to")? {
        node.floats[1] = v;
    }
    if let Some(v) = table_f32(opts, "out_from")? {
        node.floats[2] = v;
    }
    if let Some(v) = table_f32(opts, "out_to")? {
        node.floats[3] = v;
    }
    if let Some(v) = table_f32(opts, "min")? {
        node.floats[0] = v;
    }
    if let Some(v) = table_f32(opts, "max")? {
        node.floats[1] = v;
    }
    if let Some(v) = table_f32(opts, "x")? {
        node.floats[0] = v;
    }
    if let Some(v) = table_f32(opts, "y")? {
        node.floats[1] = v;
    }
    if let Some(v) = table_f32(opts, "z")? {
        node.floats[2] = v;
    }
    if let Some(v) = opts.get::<Option<u32>>("shape")? {
        node.shape = v as usize;
    }
    if let Some(Value::Table(t)) = opts.get::<Option<Value>>("floats")? {
        for i in 0..4 {
            if let Some(v) = t.get::<Option<f32>>(i + 1)? {
                node.floats[i] = v;
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct LuaDriver {
    id: u32,
}

impl UserData for LuaDriver {
    fn add_fields<F: UserDataFields<Self>>(fields: &mut F) {
        fields.add_field_method_get("id", |_, this| Ok(this.id));
        fields.add_field_method_get("name", |lua, this| {
            let state = app(lua)?;
            let name = state
                .rig
                .drivers
                .iter()
                .find(|d| d.id == this.id)
                .map(|d| d.name.clone())
                .unwrap_or_default();
            Ok(name)
        });
    }

    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method(
            "add",
            |lua, this, (kind, opts): (String, Option<Table>)| {
                let kind = DriverNodeKind::from_name(&kind).ok_or_else(|| {
                    lua_err(format!("unknown node kind '{kind}'"))
                })?;
                let state = app(lua)?;
                let node_id = {
                    let driver = state
                        .rig
                        .driver_mut(this.id)
                        .ok_or_else(|| lua_err(format!("unknown driver {}", this.id)))?;
                    let n = driver.nodes.len();
                    let mut pos = Vec2::new(
                        40.0 + (n % 4) as f32 * 220.0,
                        40.0 + (n / 4) as f32 * 160.0,
                    );
                    if let Some(ref opts) = opts {
                        if let Some(p) = table_vec2(opts, "pos")? {
                            pos = p;
                        }
                    }
                    driver.spawn_node(kind, pos)
                };
                if let Some(ref opts) = opts {
                    apply_node_opts(state, this.id, &node_id, opts)?;
                }
                Ok(node_id)
            },
        );

        methods.add_method(
            "link",
            |lua, this, (from, from_port, to, to_port): (String, String, String, String)| {
                let state = app(lua)?;
                let driver = state
                    .rig
                    .driver_mut(this.id)
                    .ok_or_else(|| lua_err(format!("unknown driver {}", this.id)))?;
                driver
                    .connect(&from, &from_port, &to, &to_port)
                    .map_err(lua_err)?;
                Ok(())
            },
        );

        methods.add_method("edit", |lua, this, ()| {
            app(lua)?.editing_driver = Some(this.id);
            Ok(())
        });

        methods.add_method("enable", |lua, this, enabled: bool| {
            let state = app(lua)?;
            let driver = state
                .rig
                .driver_mut(this.id)
                .ok_or_else(|| lua_err(format!("unknown driver {}", this.id)))?;
            driver.enabled = enabled;
            Ok(())
        });
    }
}

fn register_rig(lua: &Lua) -> mlua::Result<()> {
    let rig = lua.create_table()?;

    rig.set(
        "load_model",
        lua.create_function(|lua, path: String| {
            let resolved = resolve_path(lua, &path)?;
            let state = app(lua)?;
            let msg = state
                .load_model(&resolved)
                .map_err(|e| lua_err(e))?;
            state.status = msg.clone();
            Ok(msg)
        })?,
    )?;

    rig.set(
        "set_mode",
        lua.create_function(|lua, name: String| {
            let mode = AppMode::from_name(&name)
                .ok_or_else(|| lua_err(format!("unknown mode '{name}' (edit|pose|shape)")))?;
            app(lua)?.set_mode(mode);
            Ok(())
        })?,
    )?;

    rig.set(
        "list_bones",
        lua.create_function(|lua, ()| {
            let state = app(lua)?;
            let t = lua.create_table()?;
            for (i, b) in state.rig.bones.iter().enumerate() {
                t.set(i + 1, b.name.as_str())?;
            }
            Ok(t)
        })?,
    )?;

    rig.set(
        "bone",
        lua.create_function(|lua, name: String| {
            let state = app(lua)?;
            if state.rig.bone_by_name(&name).is_some() {
                Ok(Some(name))
            } else {
                Ok(None)
            }
        })?,
    )?;

    rig.set(
        "bones",
        lua.create_function(|lua, pattern: String| {
            let state = app(lua)?;
            let t = lua.create_table()?;
            let mut i = 1;
            for b in &state.rig.bones {
                if glob_match(&pattern, &b.name) {
                    t.set(i, b.name.as_str())?;
                    i += 1;
                }
            }
            Ok(t)
        })?,
    )?;

    rig.set(
        "create_driver",
        lua.create_function(|lua, name: Option<String>| {
            let state = app(lua)?;
            let id = state.rig.create_driver();
            if let Some(name) = name.filter(|s| !s.is_empty()) {
                if let Some(d) = state.rig.driver_mut(id) {
                    d.name = name;
                }
            }
            if state.rig.mode != AppMode::Pose {
                state.set_mode(AppMode::Pose);
            }
            Ok(LuaDriver { id })
        })?,
    )?;

    rig.set(
        "create_ik",
        lua.create_function(|lua, (bone, len): (String, Option<usize>)| {
            let state = app(lua)?;
            let id = require_bone(state, &bone)?;
            let n = len.unwrap_or(2);
            state
                .rig
                .create_ik_from_tip(&mut state.scene, id, n)
                .map_err(lua_err)
        })?,
    )?;

    rig.set(
        "create_soft",
        lua.create_function(|lua, (bone, opts): (String, Option<Table>)| {
            let state = app(lua)?;
            let id = require_bone(state, &bone)?;
            let chain_id = state
                .rig
                .create_soft_from_bone(&state.scene, id)
                .map_err(lua_err)?;
            if let Some(opts) = opts {
                let chain = state
                    .rig
                    .soft_chains
                    .iter_mut()
                    .find(|c| c.id == chain_id)
                    .ok_or_else(|| lua_err(format!("unknown soft chain {chain_id}")))?;
                if let Some(v) = table_f32(&opts, "gravity")? {
                    chain.gravity = v;
                }
                if let Some(v) = table_f32(&opts, "stiffness")? {
                    chain.stiffness = v;
                }
                if let Some(v) = table_f32(&opts, "damping")? {
                    chain.damping = v;
                }
                if let Some(v) = table_f32(&opts, "inertia")? {
                    chain.inertia = v;
                }
                if let Some(v) = table_f32(&opts, "max_angle")? {
                    chain.max_angle = v.to_radians();
                }
            }
            Ok(chain_id)
        })?,
    )?;

    rig.set(
        "create_collider",
        lua.create_function(|lua, bone: String| {
            let state = app(lua)?;
            let id = require_bone(state, &bone)?;
            state
                .rig
                .create_capsule_collider(&state.scene, id)
                .map_err(lua_err)
        })?,
    )?;

    rig.set(
        "script_dir",
        lua.create_function(|lua, ()| {
            Ok(env(lua)?.script_dir.to_string_lossy().into_owned())
        })?,
    )?;

    lua.globals().set("rig", rig)?;
    Ok(())
}

fn sandbox(lua: &Lua) -> mlua::Result<()> {
    for name in [
        "os", "io", "debug", "package", "dofile", "loadfile", "load", "require",
    ] {
        lua.globals().set(name, Value::Nil)?;
    }
    lua.globals().set(
        "print",
        lua.create_function(|lua, args: Variadic<Value>| {
            let mut parts = Vec::new();
            for v in args {
                parts.push(match v {
                    Value::String(s) => s.to_str()?.to_string(),
                    other => format!("{other:?}"),
                });
            }
            let line = parts.join("\t");
            log::info!("[lua] {line}");
            app(lua)?.status = line;
            Ok(())
        })?,
    )?;
    Ok(())
}

/// Run a Lua file against `state`. Paths in the script are relative to the file.
pub fn run_file(state: &mut AppState, path: &Path) -> Result<String, String> {
    let source = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let script_dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();

    {
        let lua = Lua::new();
        lua.set_app_data(ScriptEnv {
            app: NonNull::from(&mut *state),
            script_dir,
        });
        sandbox(&lua).map_err(|e| e.to_string())?;
        register_rig(&lua).map_err(|e| e.to_string())?;

        let name = path.to_string_lossy().into_owned();
        lua.load(&source)
            .set_name(&name)
            .exec()
            .map_err(|e| e.to_string())?;
    }

    let n_drv = state.rig.drivers.len();
    let n_ik = state.rig.ik_chains.len();
    Ok(format!(
        "Script {} · {} bones · {n_drv} drivers · {n_ik} IK",
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("script"),
        state.rig.bones.len()
    ))
}

pub fn parse_script_arg() -> Result<Option<PathBuf>, String> {
    let mut args = std::env::args().skip(1);
    let mut script = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" => {
                eprintln!(
                    "model-rig [--script path.lua]\n  -s, --script   Lua macro to run at startup"
                );
                std::process::exit(0);
            }
            "-s" | "--script" => {
                let p = args.next().ok_or_else(|| "--script needs a path".to_string())?;
                script = Some(PathBuf::from(p));
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown argument '{other}' (try --help)"));
            }
            other if other.ends_with(".lua") => {
                script = Some(PathBuf::from(other));
            }
            other => {
                return Err(format!("unexpected argument '{other}' (try --help)"));
            }
        }
    }
    Ok(script)
}

fn lua_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

fn lua_ident(id: &str) -> String {
    if id
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        id.to_string()
    } else {
        let mut s: String = id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        if s.is_empty() || s.chars().next().is_some_and(|c| c.is_ascii_digit()) {
            s.insert(0, '_');
        }
        s
    }
}

fn fmt_f(v: f32) -> String {
    if !v.is_finite() {
        return "0".into();
    }
    if (v - v.round()).abs() < 1e-4 {
        format!("{}", v.round() as i32)
    } else {
        let s = format!("{v:.4}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

fn node_opts(n: &crate::driver::DriverNode, bone_name: Option<&str>) -> String {
    use crate::driver::DriverNodeKind::*;
    let mut parts = vec![format!(
        "pos = {{{}, {}}}",
        fmt_f(n.pos.x),
        fmt_f(n.pos.y)
    )];
    if let Some(name) = bone_name {
        parts.push(format!("bone = {}", lua_str(name)));
        parts.push(format!("space = {}", lua_str(n.space.slug())));
    }
    match n.kind {
        Float => parts.push(format!("value = {}", fmt_f(n.floats[0]))),
        Vec3 | QuatEuler => {
            parts.push(format!("x = {}", fmt_f(n.floats[0])));
            parts.push(format!("y = {}", fmt_f(n.floats[1])));
            parts.push(format!("z = {}", fmt_f(n.floats[2])));
        }
        Remap => {
            parts.push(format!("from = {}", fmt_f(n.floats[0])));
            parts.push(format!("to = {}", fmt_f(n.floats[1])));
        }
        MapRange => {
            parts.push(format!("in_from = {}", fmt_f(n.floats[0])));
            parts.push(format!("in_to = {}", fmt_f(n.floats[1])));
            parts.push(format!("out_from = {}", fmt_f(n.floats[2])));
            parts.push(format!("out_to = {}", fmt_f(n.floats[3])));
        }
        Clamp => {
            parts.push(format!("min = {}", fmt_f(n.floats[0])));
            parts.push(format!("max = {}", fmt_f(n.floats[1])));
        }
        QuatScale => parts.push(format!("t = {}", fmt_f(n.floats[0]))),
        MorphSet => parts.push(format!("shape = {}", n.shape)),
        _ => {}
    }
    format!("{{ {} }}", parts.join(", "))
}

/// Lua snippet that recreates `driver` (paste into a setup script).
pub fn driver_to_lua(driver: &crate::driver::Driver, bones: &[(BoneId, String)]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "local d = rig.create_driver({})\n",
        lua_str(&driver.name)
    ));
    if !driver.enabled {
        out.push_str("d:enable(false)\n");
    }

    let mut id_map: Vec<(String, String)> = Vec::new();
    let mut used = std::collections::HashSet::new();
    for n in &driver.nodes {
        let mut ident = lua_ident(&n.id);
        if !used.insert(ident.clone()) {
            ident = format!("{ident}_{}", used.len());
            used.insert(ident.clone());
        }
        id_map.push((n.id.clone(), ident));
    }

    for n in &driver.nodes {
        let ident = id_map
            .iter()
            .find(|(id, _)| id == &n.id)
            .map(|(_, i)| i.as_str())
            .unwrap_or("n");
        let bone_name = n.bone.and_then(|id| {
            bones
                .iter()
                .find(|(b, _)| *b == id)
                .map(|(_, name)| name.as_str())
        });
        if n.kind == crate::driver::DriverNodeKind::MorphSet {
            out.push_str("-- morph_set: bind the mesh in the node editor if needed\n");
        }
        out.push_str(&format!(
            "local {ident} = d:add({}, {})\n",
            lua_str(n.kind.slug()),
            node_opts(n, bone_name)
        ));
    }

    for link in &driver.space.links {
        let from = id_map
            .iter()
            .find(|(id, _)| id == &link.from_node)
            .map(|(_, i)| i.as_str())
            .unwrap_or(&link.from_node);
        let to = id_map
            .iter()
            .find(|(id, _)| id == &link.to_node)
            .map(|(_, i)| i.as_str())
            .unwrap_or(&link.to_node);
        out.push_str(&format!(
            "d:link({from}, {}, {to}, {})\n",
            lua_str(&link.from_port),
            lua_str(&link.to_port)
        ));
    }
    out
}
