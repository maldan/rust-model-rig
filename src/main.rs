mod app;
mod cpu_profile;
mod driver;
mod framework;
mod gizmo;
mod ik;
mod ik_chain;
mod pick;
mod soft_chain;
mod sculpt;
mod script;
mod rig;
mod verlet;

use app::{default_dock, RigApp};
use framework::Host;

fn main() {
    let script = match script::parse_script_arg() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    Host::<RigApp>::run(default_dock(), script);
}
