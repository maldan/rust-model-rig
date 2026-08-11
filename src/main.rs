mod app;
mod framework;
mod gizmo;
mod ik;
mod ik_chain;
mod pick;
mod rig;
mod verlet;

use app::{default_dock, RigApp};
use framework::Host;

fn main() {
    Host::<RigApp>::run(default_dock());
}
