mod app;
mod framework;
mod gizmo;
mod pick;
mod rig;

use app::{default_dock, RigApp};
use framework::Host;

fn main() {
    Host::<RigApp>::run(default_dock());
}
