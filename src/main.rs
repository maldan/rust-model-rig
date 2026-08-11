mod app;
mod framework;
mod gizmo;
mod pick;
mod rig;
mod view_gizmo;

use app::{default_dock, RigApp};
use framework::Host;

fn main() {
    Host::<RigApp>::run(default_dock());
}
