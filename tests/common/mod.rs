use std::sync::Once;

use tracing::Level;

static INIT: Once = Once::new();

/// Automatic initialization of the tracing subscriber for tests
#[ctor::ctor]
fn auto_init_subscriber() {
    INIT.call_once(|| {
        tracing_subscriber::fmt().with_max_level(Level::INFO).init();
    });
}
